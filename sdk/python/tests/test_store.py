"""Blob stores: memory and file stores with conditions, the cache layer, prefixes, and the S3 store (SigV4 against
AWS's published examples, then a local S3 server over real HTTP that re-derives every signature it receives)."""

from __future__ import annotations

import http.server
import threading
from urllib.parse import parse_qsl, unquote, urlsplit

import pytest

from thread_sdk import (
    BlobRange,
    BlobTrace,
    CachedBlobStore,
    DiskCache,
    FileBlobStore,
    MemoryBlobStore,
    MemoryCache,
    PrefixedBlobStore,
    S3BlobStore,
    ThreadError,
    open_index,
)
from thread_sdk.s3 import EMPTY_SHA256, amz_date_of, aws_encode, sign_v4

from helpers import build_gen, row


def data(n: int, seed: int = 0) -> bytes:
    return bytes((i * 31 + seed) & 0xFF for i in range(n))


@pytest.fixture(params=["memory", "file"])
def store(request, tmp_path):
    return MemoryBlobStore() if request.param == "memory" else FileBlobStore(tmp_path / "root")


def test_get_head_put_list_with_conditions(store):
    assert store.get("a") is None
    h1 = store.put("a", "hello", if_absent=True)
    assert h1.size == 5
    assert store.put("a", "again", if_absent=True) is None
    assert store.put("a", "x", if_match="nope") is None
    h2 = store.put("a", "world!", if_match=h1.etag)
    assert h2 is not None and h2.etag != h1.etag
    assert store.head("a").etag == h2.etag
    assert bytes(store.get("a", BlobRange(1, 3))) == b"orl"
    assert [bytes(p) for p in store.get_ranges("a", [BlobRange(0, 1), BlobRange(5, 1)])] == [b"w", b"!"]
    with pytest.raises(ThreadError, match="outside"):
        store.get("a", BlobRange(4, 10))
    store.put("b/1", data(3))
    store.put("b/c/2", data(3))
    store.put("bb", data(1))
    assert store.list("b/") == ["b/1", "b/c/2"]
    assert store.list("b") == ["b/1", "b/c/2", "bb"]
    assert store.list("") == ["a", "b/1", "b/c/2", "bb"]


def test_file_store_rejects_escaping_keys(tmp_path):
    fs = FileBlobStore(tmp_path)
    for bad in ("../x", "/etc/passwd", "a/../../x", ""):
        with pytest.raises(ThreadError, match="bad object key"):
            fs.get(bad)


def test_prefixed_store():
    inner = MemoryBlobStore()
    p = PrefixedBlobStore(inner, "tenants/a/")
    p.put("idx/current.json", "{}")
    assert list(inner.objects) == ["tenants/a/idx/current.json"]
    assert p.list("idx/") == ["idx/current.json"]
    assert bytes(p.get("idx/current.json")) == b"{}"


@pytest.mark.parametrize("kind", ["memory", "disk"])
def test_cached_store(tmp_path, kind):
    inner = MemoryBlobStore()
    body = data(5000)
    inner.put("idx/g/slab/00001", body)
    inner.put("idx/current.json", '{"gen":"g"}')
    cache = MemoryCache() if kind == "memory" else DiskCache(tmp_path / "c", max_bytes=12000)
    s = CachedBlobStore(inner, cache, miss_policy="whole")
    t = BlobTrace()
    # a range miss without fill goes to the origin and caches nothing
    assert bytes(s.get("idx/g/slab/00001", BlobRange(10, 20), trace=t)) == body[10:30]
    assert (t.misses, t.reads) == (1, 1)
    # a multi-range miss without fill is the origin's answer (the memory store serves several ranges)
    assert [bytes(p) for p in s.get_ranges("idx/g/slab/00001", [BlobRange(0, 1), BlobRange(9, 2)], trace=t)] == \
        [body[0:1], body[9:11]]
    assert (t.misses, t.reads) == (2, 2)
    # a fill miss reads the whole object once and returns a view into it
    v = s.get("idx/g/slab/00001", BlobRange(100, 50), trace=t, fill=True)
    assert bytes(v) == body[100:150]
    assert isinstance(v, memoryview) and len(v.obj) == 5000
    reads = t.reads
    assert bytes(s.get("idx/g/slab/00001", BlobRange(4990, 10), trace=t)) == body[4990:]
    parts = s.get_ranges("idx/g/slab/00001", [BlobRange(10, 20), BlobRange(1000, 5)], trace=t)
    assert [bytes(p) for p in parts] == [body[10:30], body[1000:1005]]
    assert t.reads == reads and t.hits == 2
    # current.json is never cached
    s.get("idx/current.json", trace=t)
    s.get("idx/current.json", trace=t)
    assert t.reads == reads + 2
    if kind == "disk":
        # the disk cache keeps at most max_bytes: a third object evicts the least recently used one
        for k in ("idx/g/slab/00002", "idx/g/slab/00003"):
            inner.put(k, data(5000, 1))
            s.get(k, fill=True)
        assert cache.held <= 12000
        assert DiskCache(tmp_path / "c", max_bytes=12000).held == cache.held


def test_concurrent_fills_share_one_read():
    class Slow(MemoryBlobStore):
        reads = 0

        def get(self, key, range=None, *, trace=None, fill=False):
            Slow.reads += 1
            threading.Event().wait(0.05)
            return super().get(key, range, trace=trace, fill=fill)

    inner = Slow()
    inner.objects["idx/x"] = (data(1000), "e")
    s = CachedBlobStore(inner, MemoryCache(), miss_policy="whole")
    out = []
    ts = [threading.Thread(target=lambda: out.append(bytes(s.get("idx/x", BlobRange(0, 10), fill=True))))
          for _ in range(8)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()
    assert out == [data(1000)[:10]] * 8
    assert Slow.reads == 1


# ---------- S3 ----------

CREDS = {"access_key_id": "AKIAIOSFODNN7EXAMPLE", "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"}


def test_sigv4_get_object_example():
    """AWS's published SigV4 example (S3 docs, "Example: GET Object" with a Range header)."""
    h = sign_v4(method="GET", path="/test.txt", headers={"range": "bytes=0-9"}, payload_hash=EMPTY_SHA256,
                host="examplebucket.s3.amazonaws.com", region="us-east-1", amz_date="20130524T000000Z", **CREDS)
    assert h["authorization"] == (
        "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41")
    assert h["x-amz-date"] == "20130524T000000Z"
    assert "host" not in h


def test_sigv4_list_example():
    h = sign_v4(method="GET", path="/", query={"max-keys": "2", "prefix": "J"}, headers={}, payload_hash=EMPTY_SHA256,
                host="examplebucket.s3.amazonaws.com", region="us-east-1", amz_date="20130524T000000Z", **CREDS)
    assert "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7" in h["authorization"]


def test_aws_encoding_and_dates():
    assert aws_encode("idx/g4/slab/00001", True) == "idx/g4/slab/00001"
    assert aws_encode("a b+c*~/d", False) == "a%20b%2Bc%2A~%2Fd"
    assert amz_date_of(1369353600) == "20130524T000000Z"


class FakeS3(http.server.BaseHTTPRequestHandler):
    """A minimal S3: re-derives each request's SigV4 signature, serves ranges, conditional puts and ListObjectsV2."""

    objects: dict[str, bytes] = {}
    calls: list[tuple[str, str, dict[str, str]]] = []
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _verify(self) -> bool:
        auth = self.headers.get("authorization", "")
        try:
            cred, signed, sig = (p.split("=", 1)[1] for p in auth[len("AWS4-HMAC-SHA256 "):].split(", "))
            key_id, date, region, _, _ = cred.split("/")
        except ValueError:
            return False
        u = urlsplit(self.path)
        hdrs = {n: self.headers[n] for n in signed.split(";") if n not in ("host", "x-amz-date", "x-amz-content-sha256",
                                                                          "x-amz-security-token")}
        want = sign_v4(method=self.command, path=u.path, query=dict(parse_qsl(u.query)), headers=hdrs,
                       payload_hash=self.headers["x-amz-content-sha256"], host=self.headers["host"], region=region,
                       access_key_id=key_id, secret_access_key="SK", amz_date=self.headers["x-amz-date"],
                       session_token=self.headers.get("x-amz-security-token"))
        return want["authorization"] == auth

    def _reply(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _key(self) -> str:
        path = unquote(urlsplit(self.path).path)
        return path[len("/bucket/"):] if path.startswith("/bucket/") else path.lstrip("/")

    def _handle(self) -> None:
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        FakeS3.calls.append((self.command, self.path, {k.lower(): v for k, v in self.headers.items()}))
        if not self._verify():
            return self._reply(403, b"bad signature")
        q = dict(parse_qsl(urlsplit(self.path).query))
        if self.command == "GET" and q.get("list-type") == "2":
            keys = sorted(k for k in self.objects if k.startswith(q.get("prefix", "")))
            start = int(q.get("continuation-token", "0"))
            page, more = keys[start : start + 2], start + 2 < len(keys)
            xml = ('<?xml version="1.0"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                   f"<IsTruncated>{'true' if more else 'false'}</IsTruncated>"
                   + (f"<NextContinuationToken>{start + 2}</NextContinuationToken>" if more else "")
                   + "".join(f"<Contents><Key>{k.replace('&', '&amp;')}</Key></Contents>" for k in page)
                   + "</ListBucketResult>")
            return self._reply(200, xml.encode())
        key = self._key()
        cur = self.objects.get(key)
        etag = lambda b: f'"{len(b)}-{b[0] if b else 0}"'  # noqa: E731
        if self.command == "PUT":
            if self.headers.get("if-none-match") == "*" and cur is not None:
                return self._reply(412)
            im = self.headers.get("if-match")
            if im and (cur is None or im != etag(cur)):
                return self._reply(412)
            self.objects[key] = body
            return self._reply(200, b"", {"etag": etag(body)})
        if cur is None:
            return self._reply(404, b"<Error><Code>NoSuchKey</Code></Error>")
        rng = self.headers.get("range")
        if rng:
            a, b = (int(x) for x in rng.removeprefix("bytes=").split("-"))
            b = min(b, len(cur) - 1)
            return self._reply(206, cur[a : b + 1], {"content-range": f"bytes {a}-{b}/{len(cur)}"})
        return self._reply(200, cur, {"etag": etag(cur)})

    do_GET = do_PUT = do_HEAD = _handle


@pytest.fixture
def s3_server():
    FakeS3.objects = {}
    FakeS3.calls = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeS3, bind_and_activate=False)
    srv.request_queue_size = 128  # open_index warms up to 32 connections at once
    srv.server_bind()
    srv.server_activate()
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()  # connections still queued are refused, not left waiting


def test_s3_store(s3_server):
    s = S3BlobStore(s3_server, "bucket", "AK", "SK")
    h1 = s.put("idx/current.json", '{"gen":"g4","rev":1}', if_absent=True, content_type="application/json")
    assert h1 is not None and h1.etag == "20-123"  # quotes stripped
    assert s.put("idx/current.json", "x", if_absent=True) is None
    assert s.put("idx/current.json", "x", if_match="wrong") is None
    assert s.put("idx/current.json", '{"gen":"g4","rev":2}', if_match=h1.etag) is not None
    assert s.head("idx/current.json").size == 20
    assert s.head("idx/nope") is None
    assert s.get("idx/nope") is None
    body = bytes(i & 0xFF for i in range(300))
    s.put("idx/g4/slab/00001", body)
    s.put("idx/g4/a b+c.json", b"odd key")
    assert s.get("idx/g4/slab/00001") == body
    assert s.get("idx/g4/slab/00001", BlobRange(100, 50)) == body[100:150]
    assert s.get("idx/g4/a b+c.json") == b"odd key"
    assert s.list("idx/") == ["idx/current.json", "idx/g4/a b+c.json", "idx/g4/slab/00001"]  # paginated, 2 per page
    assert s.get_ranges("idx/g4/slab/00001", [BlobRange(0, 1), BlobRange(5, 1)]) is None  # no multi-range in S3
    put = next(c for c in FakeS3.calls if c[0] == "PUT")
    assert put[1] == "/bucket/idx/current.json"
    assert put[2]["if-none-match"] == "*"
    get = next(c for c in FakeS3.calls if c[0] == "GET" and "range" in c[2])
    assert "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date" in get[2]["authorization"]
    tok = S3BlobStore(s3_server, "bucket", "AK", "SK", session_token="tok", region="us-east-1")
    assert tok.get("idx/current.json") is not None
    assert FakeS3.calls[-1][2]["x-amz-security-token"] == "tok"
    bad = S3BlobStore(s3_server, "bucket", "AK", "wrong-secret")
    with pytest.raises(ThreadError, match="403"):
        bad.get("idx/current.json")


def test_index_over_s3(s3_server):
    """A generation uploaded through S3, queried and updated through S3 with the default memory cache."""
    b = build_gen("gs", 3)
    s = S3BlobStore(s3_server, "bucket", "AK", "SK")
    for k in b.store.list(""):
        s.put(k, b.store.get(k))
    with open_index(b.store, cache=False) as ref, open_index(s) as ix:
        q = row(b.vectors, 21, 16)
        want = [(h.id, h.score) for h in ref.query(q, nprobe=8).hits]
        r1 = ix.query(q, nprobe=8)
        assert [(h.id, h.score) for h in r1.hits] == want
        assert r1.stats["io"]["warms"] == 8  # "range+warm": the probed slabs are fetched whole in the background
        assert ix.store.drain(5)
        r2 = ix.query(q, nprobe=8)
        assert r2.stats["io"]["reads"] == 0  # served by the memory cache
        ix.upsert([777], q)
        assert ix.query(q, top_k=2).hits[0].id in (21, 777)
        assert {h.id for h in ix.query(q, top_k=2).hits} == {21, 777}
    with open_index(s, cache=False) as nocache:
        r = nocache.query(q, nprobe=8, rerank=40)
        assert r.stats["rerank_reads"] > 0  # single-range reads in parallel, no multi-range on S3
        assert {h.id for h in r.hits[:2]} == {21, 777}
