"""`S3BlobStore`: the BlobStore interface over any S3-compatible endpoint (R2's S3 API, AWS S3, MinIO, ...) with the
standard library only: `http.client` with one keep-alive connection per thread and host, and AWS Signature V4.
Single-range GETs, HEAD, PUT and conditional PUT (`If-None-Match: *` for create-only, `If-Match` for compare-and-
swap) and ListObjectsV2. S3 has no multi-range GET, so `get_ranges` is not implemented: the SDK issues one request
per rerank range (in parallel), and a cache in front (`open` adds a `MemoryCache` by default) serves repeat reads.

Connections: a pool of keep-alive `http.client` connections shared by every thread (at most `max_connections`, default
`inflight`; a thread takes an idle one or opens one, and waits when all are busy), reopened when the server closed
them; `warmup(n)` opens them ahead of the first query. Signing
costs one HMAC per request: the derived signing key of each (date, region, service) is cached.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Executor, Future
from typing import Callable, Mapping
from urllib.parse import quote, urlsplit

from .store import BlobHead, BlobStore, _encode, _trace_read
from .wire import ThreadError

__all__ = ["S3BlobStore", "SigningKeyCache", "sign_v4", "derive_signing_key", "aws_encode", "amz_date_of",
           "EMPTY_SHA256", "S3_INFLIGHT"]

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
#: Reads in flight per query against S3 directly (no per-process connection cap).
S3_INFLIGHT = 32


def aws_encode(s: str, keep_slash: bool) -> str:
    """RFC 3986 encoding, AWS style (everything but `A-Za-z0-9-_.~` escaped; `/` kept in paths)."""
    return quote(s, safe="-_.~/" if keep_slash else "-_.~")


def amz_date_of(seconds: float) -> str:
    """`YYYYMMDDTHHMMSSZ`."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(seconds))


def _hmac(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def derive_signing_key(secret_access_key: str, date: str, region: str, service: str) -> bytes:
    """The SigV4 signing key `kSigning` for (date, region, service) (four HMACs)."""
    k = _hmac(("AWS4" + secret_access_key).encode("utf-8"), date)
    for part in (region, service, "aws4_request"):
        k = _hmac(k, part)
    return k


class SigningKeyCache:
    """Derived signing keys of one secret by (date, region, service): a request then costs one HMAC instead of five.
    Holds the last few scopes (the date rolls daily). Thread-safe."""

    def __init__(self, secret_access_key: str) -> None:
        self._secret = secret_access_key
        self._keys: dict[tuple[str, str, str], bytes] = {}
        self._lock = threading.Lock()
        #: Key derivations performed (tests, benchmarks).
        self.derived = 0

    def get(self, date: str, region: str, service: str) -> bytes:
        scope = (date, region, service)
        k = self._keys.get(scope)
        if k is None:
            k = derive_signing_key(self._secret, date, region, service)
            with self._lock:
                self.derived += 1
                if len(self._keys) >= 4:
                    self._keys.pop(next(iter(self._keys)))
                self._keys[scope] = k
        return k


def sign_v4(
    *,
    method: str,
    path: str,
    headers: Mapping[str, str],
    payload_hash: str,
    host: str,
    region: str,
    access_key_id: str,
    secret_access_key: str,
    amz_date: str,
    query: Mapping[str, str] | None = None,
    session_token: str | None = None,
    service: str = "s3",
    keys: SigningKeyCache | None = None,
) -> dict[str, str]:
    """Signs a request (AWS Signature Version 4, header auth). `path` is the already-encoded canonical path,
    `query` unencoded, `headers` the extra headers to sign (lower-case names). Returns the headers to send (without
    `host`, which the connection sets). With `keys` (a cache for `secret_access_key`) the derived signing key is
    reused across requests."""
    date = amz_date[:8]
    h = {**headers, "host": host, "x-amz-date": amz_date, "x-amz-content-sha256": payload_hash}
    if session_token:
        h["x-amz-security-token"] = session_token
    names = sorted(h)
    canonical_headers = "".join(f"{n}:{' '.join(h[n].split())}\n" for n in names)
    signed = ";".join(names)
    q = "&".join(f"{k}={v}" for k, v in sorted((aws_encode(k, False), aws_encode(v, False)) for k, v in (query or {}).items()))
    canonical = "\n".join([method, path, q, canonical_headers, signed, payload_hash])
    scope = f"{date}/{region}/{service}/aws4_request"
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode("utf-8")).hexdigest()])
    k = keys.get(date, region, service) if keys is not None else derive_signing_key(secret_access_key, date, region,
                                                                                    service)
    signature = hmac.new(k, to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    out = {n: v for n, v in h.items() if n != "host"}
    out["authorization"] = f"AWS4-HMAC-SHA256 Credential={access_key_id}/{scope}, SignedHeaders={signed}, Signature={signature}"
    return out


def _strip_quotes(etag: str | None) -> str:
    e = etag or ""
    if e.startswith("W/"):
        e = e[2:]
    return e.strip('"')


class _Response:
    __slots__ = ("status", "headers", "body")

    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body


_RETRYABLE = (ConnectionError, http.client.RemoteDisconnected, http.client.BadStatusLine, TimeoutError, OSError)


class S3BlobStore(BlobStore):
    """`endpoint`: `https://<account>.r2.cloudflarestorage.com`, `https://s3.us-east-1.amazonaws.com`,
    `http://localhost:9000`, ...; `region`: the signing region (R2: `auto`); `path_style`: `endpoint/bucket/key`
    (default, works everywhere) vs `bucket.endpoint/key`."""

    supports_put = True
    supports_list = True
    remote = True
    inflight = S3_INFLIGHT

    def __init__(
        self,
        endpoint: str,
        bucket: str,
        access_key_id: str,
        secret_access_key: str,
        *,
        region: str = "auto",
        session_token: str | None = None,
        path_style: bool = True,
        retries: int = 1,
        timeout: float = 30.0,
        now: Callable[[], float] = time.time,
        inflight: int | None = None,
        coalesce_bytes: int | None = None,
        span_bytes: int | None = None,
        max_connections: int | None = None,
    ) -> None:
        if not access_key_id or not secret_access_key:
            raise ThreadError("S3BlobStore needs access_key_id and secret_access_key", 400)
        u = urlsplit(endpoint)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ThreadError(f"bad S3 endpoint {endpoint!r}", 400)
        self.scheme = u.scheme
        self.endpoint_host = u.netloc
        self.base_path = u.path.rstrip("/")
        self.bucket = bucket
        self.region = region
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.path_style = path_style
        self.retries = retries
        self.timeout = timeout
        self.now = now
        #: Most connections open at once (shared by all threads); default: `inflight`.
        self.max_connections = max_connections if max_connections is not None else (inflight or S3_INFLIGHT)
        self._idle: dict[str, list[http.client.HTTPConnection]] = {}
        self._open = 0
        self._pool_cond = threading.Condition()
        #: Connections opened over the store's life (tests, benchmarks).
        self.stats = {"opened": 0}
        self.signing_keys = SigningKeyCache(secret_access_key)
        if inflight is not None:
            self.inflight = inflight
        self.coalesce_bytes = coalesce_bytes
        self.span_bytes = span_bytes

    @classmethod
    def r2(cls, account_id: str, bucket: str, access_key_id: str, secret_access_key: str, **kw) -> "S3BlobStore":
        """R2's S3 API for an account (`https://<account_id>.r2.cloudflarestorage.com`, region `auto`)."""
        return cls(f"https://{account_id}.r2.cloudflarestorage.com", bucket, access_key_id, secret_access_key, **kw)

    def _target(self, key: str) -> tuple[str, str]:
        """Host and canonical path of an object (or of the bucket when `key` is empty)."""
        if self.path_style:
            host = self.endpoint_host
            path = f"{self.base_path}/{aws_encode(self.bucket, False)}" + (f"/{aws_encode(key, True)}" if key else "")
        else:
            host = f"{self.bucket}.{self.endpoint_host}"
            path = f"{self.base_path}/{aws_encode(key, True) if key else ''}"
        return host, path or "/"

    def _acquire(self, host: str, fresh: bool) -> http.client.HTTPConnection:
        """A connection to `host` from the shared pool: an idle keep-alive one (unless `fresh`), else a new one while
        fewer than `max_connections` are open, else the next one released (waits)."""
        with self._pool_cond:
            while True:
                idle = self._idle.setdefault(host, [])
                if idle and not fresh:
                    return idle.pop()
                if self._open < self.max_connections:
                    self._open += 1
                    self.stats["opened"] += 1
                    break
                if idle:  # fresh wanted, pool full: replace an idle connection
                    idle.pop().close()
                    self.stats["opened"] += 1
                    break
                self._pool_cond.wait()
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(host, timeout=self.timeout)

    def _release(self, host: str, conn: http.client.HTTPConnection, reuse: bool) -> None:
        with self._pool_cond:
            if reuse:
                self._idle.setdefault(host, []).append(conn)
            else:
                conn.close()
                self._open -= 1
            self._pool_cond.notify()

    def _send(self, method: str, key: str, headers: dict[str, str], body: bytes | None = None,
              query: dict[str, str] | None = None) -> _Response:
        host, path = self._target(key)
        payload_hash = hashlib.sha256(body).hexdigest() if body is not None else EMPTY_SHA256
        signed = sign_v4(method=method, path=path, query=query, headers=headers, payload_hash=payload_hash, host=host,
                         region=self.region, access_key_id=self.access_key_id,
                         secret_access_key=self.secret_access_key, session_token=self.session_token,
                         amz_date=amz_date_of(self.now()), keys=self.signing_keys)
        qs = "?" + "&".join(f"{aws_encode(k, False)}={aws_encode(v, False)}" for k, v in query.items()) if query else ""
        # A PUT is never retried (a conditional put that reached the server must not be repeated) and goes over a
        # fresh connection, so a keep-alive connection the server dropped cannot fail it.
        attempts = 1 if method == "PUT" else 1 + self.retries
        last: BaseException | None = None
        for attempt in range(attempts):
            conn = self._acquire(host, fresh=method == "PUT" or attempt > 0)
            try:
                conn.request(method, path + qs, body=body, headers=signed)
                r = conn.getresponse()
                data = r.read()
                hdrs = {k.lower(): v for k, v in r.getheaders()}
            except _RETRYABLE as e:
                self._release(host, conn, False)
                last = e
                continue
            except BaseException:
                self._release(host, conn, False)
                raise
            self._release(host, conn, not r.will_close)
            if r.status >= 500 and attempt + 1 < attempts:
                last = ThreadError(f"S3 {method} {key}: HTTP {r.status}", r.status)
                continue
            return _Response(r.status, hdrs, data)
        if isinstance(last, ThreadError):
            raise last
        raise ThreadError(f"S3 {method} {key} failed: {last!r}", 502) from last

    @staticmethod
    def _fail(r: _Response, what: str) -> ThreadError:
        text = r.body[:300].decode("utf-8", "replace")
        return ThreadError(f"S3 {what}: HTTP {r.status} {text}", 403 if r.status == 403 else 502)

    def get(self, key, range=None, *, trace=None, fill=False):
        headers = {"range": f"bytes={range.offset}-{range.offset + range.length - 1}"} if range else {}
        if range is not None and range.length == 0:
            return b""
        r = self._send("GET", key, headers)
        if r.status == 404:
            return None
        if not 200 <= r.status < 300:
            raise self._fail(r, f"GET {key}")
        b = r.body
        _trace_read(trace, len(b))
        if range is not None:
            if r.status == 200:  # Range ignored
                b = b[range.offset : range.offset + range.length]
            if len(b) != range.length:
                raise ThreadError(f"{key} [{range.offset}, +{range.length}): got {len(b)} bytes")
        return b

    def warmup(self, n: int, executor: Executor | None = None, key: str = "idx/current.json") -> list[Future]:
        """Opens up to `n` connections ahead of the first query: `n` concurrent HEADs of `key` (a 404 is fine); they
        return to the shared pool, so any thread's later read reuses them. Runs on `executor` (`open_index` passes its
        I/O pool) or on new threads. Returns the futures (errors are swallowed)."""

        def one() -> None:
            try:
                self.head(key)
            except Exception:  # noqa: BLE001 - warm-up is best effort
                pass

        if n <= 0:
            return []
        if executor is not None:
            return [executor.submit(one) for _ in range(n)]
        out = []
        for _ in range(n):
            f: Future = Future()
            def run(f: Future = f) -> None:
                one()
                f.set_result(None)

            t = threading.Thread(target=run, daemon=True)
            t.start()
            out.append(f)
        return out

    def head(self, key):
        r = self._send("HEAD", key, {})
        if r.status == 404:
            return None
        if not 200 <= r.status < 300:
            raise ThreadError(f"S3 HEAD {key}: HTTP {r.status}", 502)
        return BlobHead(int(r.headers.get("content-length", 0)), _strip_quotes(r.headers.get("etag")))

    def put(self, key, data, *, if_absent=False, if_match=None, content_type=None):
        """Conditional puts use `If-None-Match: *` (create-only) and `If-Match: "<etag>"`; a 412 (or 409) means the
        condition failed. R2's S3 API and AWS S3 support both on PutObject."""
        body = _encode(data)
        headers = {"content-type": content_type or "application/octet-stream", "content-length": str(len(body))}
        if if_absent:
            headers["if-none-match"] = "*"
        if if_match is not None:
            headers["if-match"] = f'"{_strip_quotes(if_match)}"'
        r = self._send("PUT", key, headers, body)
        if r.status in (412, 409):
            return None
        if not 200 <= r.status < 300:
            raise self._fail(r, f"PUT {key}")
        return BlobHead(len(body), _strip_quotes(r.headers.get("etag")))

    def list(self, prefix):
        out: list[str] = []
        token: str | None = None
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token:
                query["continuation-token"] = token
            r = self._send("GET", "", {}, None, query)
            if not 200 <= r.status < 300:
                raise self._fail(r, f"LIST {prefix}")
            root = ET.fromstring(r.body)
            out.extend(k.text or "" for k in root.iterfind(".//{*}Key"))
            truncated = (root.findtext("{*}IsTruncated") or "").strip() == "true"
            token = root.findtext("{*}NextContinuationToken")
            if not truncated or not token:
                return out
