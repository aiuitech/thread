"""Embedding models: the `embedding` block of a manifest names the model a dense generation was built for, and
`resolve_embedder` turns it into a function the index calls for query texts (`search(text=...)` without a vector)
and document texts (`upsert(texts=...)` without vectors). The SDK never runs a model itself: `workers-ai` calls the
Workers AI REST API, `openai` any OpenAI-compatible `/embeddings` endpoint, and `custom` whatever callable the
caller passes to `open_index(embed=...)`. The config shape, the prefixing rule and the batching are shared with the
TypeScript SDK through `sdk/ts/test/fixtures/embedding.json`.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from array import array
from typing import Any, Callable, Literal, Sequence

from .wire import ThreadError

__all__ = [
    "EMBEDDING_PROVIDERS",
    "EMBED_BATCH_DEFAULTS",
    "EmbedKind",
    "validate_embedding_config",
    "embedding_inputs",
    "embedding_batch_size",
    "embedding_batches",
    "resolve_embedder",
    "split_rows",
    "normalize_rows",
]

#: Providers the SDKs call themselves; any other `provider` needs a caller-supplied embedder.
EMBEDDING_PROVIDERS: tuple[str, ...] = ("workers-ai", "openai", "custom")
#: Texts per provider call when the config does not say (Workers AI accepts 100 strings per request).
EMBED_BATCH_DEFAULTS: dict[str, int] = {"workers-ai": 100, "openai": 2048, "custom": 256}

#: What a text is embedded as: a query (search) or a document (upsert). Picks the prefix.
EmbedKind = Literal["query", "document"]
#: `(url, headers, body) -> (status, body)`: the HTTP transport (`urllib` by default; tests pass their own).
HttpPost = Callable[[str, dict[str, str], bytes], tuple[int, bytes]]
#: One `array("f")` row per text.
Rows = list[array]


def validate_embedding_config(c: Any, dim: int | None = None) -> dict[str, Any]:
    """Checks an `embedding` block and returns it with unknown fields dropped. `dim` (the generation's) must match
    `config["dim"]` when given."""
    if not isinstance(c, dict):
        raise ThreadError("embedding must be an object {provider, model, dim, ...}", 400)
    provider = c.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ThreadError("embedding.provider must be a non-empty string", 400)
    if provider not in EMBEDDING_PROVIDERS:
        raise ThreadError(f"embedding.provider must be one of {', '.join(EMBEDDING_PROVIDERS)}", 400)
    model = c.get("model")
    if not isinstance(model, str) or not model:
        raise ThreadError("embedding.model must be a non-empty string", 400)
    d = c.get("dim")
    if isinstance(d, bool) or not isinstance(d, int) or d < 1:
        raise ThreadError("embedding.dim must be an integer >= 1", 400)
    if dim is not None and d != dim:
        raise ThreadError(f"embedding.dim {d} does not match the generation's dim {dim}", 400)
    for f in ("queryPrefix", "docPrefix", "baseUrl"):
        if c.get(f) is not None and not isinstance(c[f], str):
            raise ThreadError(f"embedding.{f} must be a string", 400)
    options = c.get("options")
    if options is not None and not isinstance(options, dict):
        raise ThreadError("embedding.options must be an object", 400)
    mb = c.get("maxBatch")
    if mb is not None and (isinstance(mb, bool) or not isinstance(mb, int) or mb < 1):
        raise ThreadError("embedding.maxBatch must be an integer >= 1", 400)
    out: dict[str, Any] = {"provider": provider, "model": model, "dim": d}
    for f in ("queryPrefix", "docPrefix", "baseUrl"):
        if c.get(f):
            out[f] = c[f]
    if options:
        out["options"] = dict(options)
    if mb is not None:
        out["maxBatch"] = mb
    return out


def embedding_inputs(c: dict[str, Any], texts: Sequence[str], kind: str) -> list[str]:
    """The strings sent to the model for `texts` of `kind`: the kind's prefix in front of each text."""
    prefix = (c.get("queryPrefix") if kind == "query" else c.get("docPrefix")) or ""
    return [prefix + t for t in texts]


def embedding_batch_size(c: dict[str, Any]) -> int:
    """Texts per provider call for a config."""
    return int(c.get("maxBatch") or EMBED_BATCH_DEFAULTS.get(c["provider"], 256))


def embedding_batches(c: dict[str, Any], n: int) -> list[int]:
    """Sizes of the provider calls that embed `n` texts under `c`."""
    size = embedding_batch_size(c)
    return [min(size, n - i) for i in range(0, n, size)]


def split_rows(v: Any, n: int, dim: int, what: str) -> Rows:
    """One `array("f")` row per text from what an embedder returned: flat `n * dim` floats, one row per text, or an
    `(n, dim)` numpy array."""
    if hasattr(v, "shape") and hasattr(v, "tolist"):
        shape = tuple(v.shape)
        if len(shape) == 2:
            if shape[0] != n or shape[1] != dim:
                raise ThreadError(f"{what} returned an array of shape {shape}, expected ({n}, {dim})", 400)
            return [array("f", r) for r in v.tolist()]
        v = v.tolist() if len(shape) == 1 else v.reshape(-1).tolist()
    if isinstance(v, (bytes, bytearray, memoryview)):
        flat = array("f")
        flat.frombytes(bytes(v))
        v = flat
    seq = list(v) if not isinstance(v, (list, tuple, array)) else v
    if len(seq) and not isinstance(seq[0], (int, float)):
        if len(seq) != n:
            raise ThreadError(f"{what} returned {len(seq)} vectors for {n} texts", 400)
        rows: Rows = []
        for i, r in enumerate(seq):
            row = array("f", r.tolist() if hasattr(r, "tolist") else r)
            if len(row) != dim:
                raise ThreadError(f"{what} returned vector {i} with {len(row)} dims, expected {dim}", 400)
            rows.append(row)
        return rows
    if len(seq) != n * dim:
        raise ThreadError(f"{what} returned {len(seq)} floats for {n} texts of {dim} dims", 400)
    flat_all = array("f", seq) if not isinstance(seq, array) else seq
    return [flat_all[i * dim:(i + 1) * dim] for i in range(n)]


def normalize_rows(rows: Rows, what: str) -> Rows:
    """Unit-normalizes rows in place (a zero or non-finite row is an error)."""
    for i, r in enumerate(rows):
        ss = sum(x * x for x in r)
        if not ss > 0 or not math.isfinite(ss):
            raise ThreadError(f"{what}: vector {i} has zero or non-finite norm", 400)
        inv = 1 / math.sqrt(ss)
        for j in range(len(r)):
            r[j] *= inv
    return rows


def _rows_of(data: Any, n: int, dim: int, what: str) -> Rows:
    """Rows of a provider response (`data: [[...]]`, or OpenAI's `data: [{embedding, index}]`) checked against `dim`."""
    if not isinstance(data, list) or len(data) != n:
        raise ThreadError(f"{what} returned {len(data) if isinstance(data, list) else 'no'} embeddings for {n} texts", 502)
    rows: list[array | None] = [None] * n
    for i, row in enumerate(data):
        v: Any = row
        at = i
        if isinstance(row, dict):
            v = row.get("embedding")
            if isinstance(row.get("index"), int) and not isinstance(row.get("index"), bool):
                at = row["index"]
        if not isinstance(v, list) or len(v) != dim:
            raise ThreadError(f"{what} returned a vector of {len(v) if isinstance(v, list) else 'no'} dims, expected {dim}", 502)
        if at < 0 or at >= n or rows[at] is not None:
            raise ThreadError(f"{what} returned a bad embedding index {at}", 502)
        rows[at] = array("f", v)
    return [r for r in rows if r is not None]


def _urllib_post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - the provider URL from the config
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def resolve_embedder(c: dict[str, Any], *, embedder: Callable[[list[str], str], Any] | None = None,
                     account_id: str | None = None, api_token: str | None = None, api_key: str | None = None,
                     headers: dict[str, str] | None = None, post: HttpPost | None = None
                     ) -> Callable[[Sequence[str], str], Rows]:
    """The embedder for a config: `provider: "workers-ai"` POSTs to the Workers AI REST API with `account_id` and
    `api_token`; `"openai"` to `{baseUrl}/embeddings` with `api_key`; `"custom"` (or any provider with `embedder`)
    calls `embedder(texts, kind)`. The returned function takes the texts of one kind, prefixes them, calls the
    provider in batches of `maxBatch` and returns one `array("f")` row per text (not yet normalized)."""
    size = embedding_batch_size(c)
    dim = int(c["dim"])
    model = str(c["model"])
    send = post or _urllib_post

    def http(url: str, hdrs: dict[str, str], payload: dict[str, Any], what: str) -> Any:
        status, body = send(url, {"content-type": "application/json", **hdrs, **(headers or {})}, json.dumps(payload).encode())
        if status < 200 or status >= 300:
            raise ThreadError(f"{what}: HTTP {status}: {body[:300].decode('utf-8', 'replace')}", 502)
        try:
            return json.loads(body)
        except ValueError:
            raise ThreadError(f"{what}: non-JSON response", 502) from None

    call: Callable[[list[str], str], Rows]
    if embedder is not None:
        custom = embedder

        def call(inputs: list[str], kind: str) -> Rows:
            return split_rows(custom(inputs, kind), len(inputs), dim, "the custom embedder")
    elif c["provider"] == "workers-ai":
        if not account_id or not api_token:
            raise ThreadError(f"embedding model {model}: pass account_id and api_token for Workers AI", 400)
        base = str(c.get("baseUrl") or "https://api.cloudflare.com/client/v4").rstrip("/")
        url = f"{base}/accounts/{account_id}/ai/run/{model}"
        what = f"Workers AI {model}"

        def call(inputs: list[str], kind: str) -> Rows:
            j = http(url, {"authorization": f"Bearer {api_token}"}, {**(c.get("options") or {}), "text": inputs}, what)
            return _rows_of((j.get("result") or {}).get("data") if isinstance(j, dict) else None, len(inputs), dim, what)
    elif c["provider"] == "openai":
        base = str(c.get("baseUrl") or "https://api.openai.com/v1").rstrip("/")
        what = f"{model} at {base}"
        auth = {"authorization": f"Bearer {api_key}"} if api_key else {}

        def call(inputs: list[str], kind: str) -> Rows:
            j = http(f"{base}/embeddings", auth, {**(c.get("options") or {}), "model": model, "input": inputs}, what)
            return _rows_of(j.get("data") if isinstance(j, dict) else None, len(inputs), dim, what)
    else:
        raise ThreadError(f"embedding provider {c['provider']} needs an embedder: open_index(embed=lambda texts, kind: ...)", 400)

    def embed(texts: Sequence[str], kind: str) -> Rows:
        inputs = embedding_inputs(c, texts, kind)
        out: Rows = []
        for i in range(0, len(inputs), size):
            out.extend(call(inputs[i:i + size], kind))
        return out

    return embed
