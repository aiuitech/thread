"""`S3BlobStore`: the BlobStore interface over any S3-compatible endpoint (R2's S3 API, AWS S3, MinIO, ...) with the
standard library only: `http.client` with one keep-alive connection per thread and host, and AWS Signature V4.
Single-range GETs, HEAD, PUT and conditional PUT (`If-None-Match: *` for create-only, `If-Match` for compare-and-
swap) and ListObjectsV2. S3 has no multi-range GET, so `get_ranges` is not implemented: the SDK issues one request
per rerank range (in parallel), and a cache in front (`open` adds a `MemoryCache` by default) serves repeat reads.
"""

from __future__ import annotations

import hashlib
import hmac
import http.client
import threading
import time
import xml.etree.ElementTree as ET
from typing import Callable, Mapping
from urllib.parse import quote, urlsplit

from .store import BlobHead, BlobStore, _encode, _trace_read
from .wire import ThreadError

__all__ = ["S3BlobStore", "sign_v4", "aws_encode", "amz_date_of", "EMPTY_SHA256"]

EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def aws_encode(s: str, keep_slash: bool) -> str:
    """RFC 3986 encoding, AWS style (everything but `A-Za-z0-9-_.~` escaped; `/` kept in paths)."""
    return quote(s, safe="-_.~/" if keep_slash else "-_.~")


def amz_date_of(seconds: float) -> str:
    """`YYYYMMDDTHHMMSSZ`."""
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(seconds))


def _hmac(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


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
) -> dict[str, str]:
    """Signs a request (AWS Signature Version 4, header auth). `path` is the already-encoded canonical path,
    `query` unencoded, `headers` the extra headers to sign (lower-case names). Returns the headers to send (without
    `host`, which the connection sets)."""
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
    k = _hmac(("AWS4" + secret_access_key).encode("utf-8"), date)
    for part in (region, service, "aws4_request"):
        k = _hmac(k, part)
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
        self._local = threading.local()

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

    def _conn(self, host: str, fresh: bool) -> http.client.HTTPConnection:
        conns: dict[str, http.client.HTTPConnection] = self._local.__dict__.setdefault("conns", {})
        c = conns.get(host)
        if c is not None and fresh:
            c.close()
            c = None
        if c is None:
            cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
            c = cls(host, timeout=self.timeout)
            conns[host] = c
        return c

    def _drop(self, host: str) -> None:
        conns = self._local.__dict__.get("conns", {})
        c = conns.pop(host, None)
        if c is not None:
            c.close()

    def _send(self, method: str, key: str, headers: dict[str, str], body: bytes | None = None,
              query: dict[str, str] | None = None) -> _Response:
        host, path = self._target(key)
        payload_hash = hashlib.sha256(body).hexdigest() if body is not None else EMPTY_SHA256
        signed = sign_v4(method=method, path=path, query=query, headers=headers, payload_hash=payload_hash, host=host,
                         region=self.region, access_key_id=self.access_key_id,
                         secret_access_key=self.secret_access_key, session_token=self.session_token,
                         amz_date=amz_date_of(self.now()))
        qs = "?" + "&".join(f"{aws_encode(k, False)}={aws_encode(v, False)}" for k, v in query.items()) if query else ""
        # A PUT is never retried (a conditional put that reached the server must not be repeated) and goes over a
        # fresh connection, so a keep-alive connection the server dropped cannot fail it.
        attempts = 1 if method == "PUT" else 1 + self.retries
        last: BaseException | None = None
        for attempt in range(attempts):
            conn = self._conn(host, fresh=method == "PUT" or attempt > 0)
            try:
                conn.request(method, path + qs, body=body, headers=signed)
                r = conn.getresponse()
                data = r.read()
                hdrs = {k.lower(): v for k, v in r.getheaders()}
                if r.will_close:
                    self._drop(host)
            except _RETRYABLE as e:
                self._drop(host)
                last = e
                continue
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
