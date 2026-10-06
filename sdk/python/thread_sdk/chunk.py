"""Flat word-window chunking of a text before it is embedded and upserted: `chunk_text` splits a text into windows
of `words` words (default 200) that share `overlap` words with the previous window (default 0), each a verbatim
slice of the original text. Chunking is a client-side transform (the index never sees the parent text), so the
TypeScript SDK (`chunkText`) and the CLI (`thread chunk`) implement the same rule; the three are checked against
sdk/ts/test/fixtures/chunks.json.

A word is a maximal run of characters that are not Unicode White_Space (what Rust's `char::is_whitespace` tests:
`\\t \\n \\v \\f \\r`, space, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000). Chunk
`i` holds words `[i * (words - overlap), i * (words - overlap) + words)`; the last chunk is whatever remains and no
chunk starts once the previous one reached the end. A text with no words is one empty chunk, so a keyed document
always writes at least one chunk. `start` / `end` are code-point offsets of the chunk in the text (`text[start:end]`
is the chunk; whitespace inside a chunk is kept, whitespace around it is not).

A chunked upsert (`index.upsert(..., chunk=True)`) stores chunk `i` of the document keyed `k` under the key
`chunk_key(k, i)` = `f"{k}#{i}"`, with the parent's meta and attrs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

__all__ = ["DEFAULT_CHUNK_WORDS", "DEFAULT_CHUNK_OVERLAP", "CHUNK_KEY_SEPARATOR", "Chunk", "ChunkOptions",
           "chunk_text", "chunk_key", "chunk_index_of", "resolve_chunk_options"]

DEFAULT_CHUNK_WORDS = 200
DEFAULT_CHUNK_OVERLAP = 0
#: Between a document's key and its chunk index in a chunk's key: `f"{key}#{i}"`.
CHUNK_KEY_SEPARATOR = "#"

# Unicode White_Space, spelled out so that every implementation agrees (Python's `str.split` and JS's `\s` each
# include a few characters the other does not).
_WORD = re.compile("[^\t\n\x0b\x0c\r \x85\xa0  -     　]+")
_INDEX = re.compile(r"[0-9]+")


@dataclass(frozen=True)
class Chunk:
    """One chunk: its position among the text's chunks, its text, its code-point span in the text and its word count."""

    index: int
    text: str
    start: int
    end: int
    words: int


@dataclass(frozen=True)
class ChunkOptions:
    """Flat chunking parameters: `words` per chunk (>= 1) and `overlap` words shared with the previous chunk
    (0 <= overlap < words)."""

    words: int = DEFAULT_CHUNK_WORDS
    overlap: int = DEFAULT_CHUNK_OVERLAP


def resolve_chunk_options(opts: ChunkOptions | Mapping[str, Any] | bool | None) -> ChunkOptions:
    """`True` / `None` -> the defaults; a dict `{"words": 200, "overlap": 0}` or a `ChunkOptions`; validated."""
    if opts is None or opts is True:
        o = ChunkOptions()
    elif isinstance(opts, ChunkOptions):
        o = opts
    elif isinstance(opts, Mapping):
        extra = set(opts) - {"words", "overlap"}
        if extra:
            raise ValueError(f"unknown chunk option(s): {', '.join(sorted(extra))}")
        o = ChunkOptions(opts.get("words", DEFAULT_CHUNK_WORDS), opts.get("overlap", DEFAULT_CHUNK_OVERLAP))
    else:
        raise ValueError("chunk must be True, a dict {words, overlap} or ChunkOptions")
    for name, v in (("words", o.words), ("overlap", o.overlap)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"chunk {name} must be an integer, got {v!r}")
    if o.words < 1:
        raise ValueError(f"chunk words must be >= 1, got {o.words}")
    if not 0 <= o.overlap < o.words:
        raise ValueError(f"chunk overlap must be in [0, words), got {o.overlap} for {o.words} words")
    return o


def chunk_text(text: str, words: int = DEFAULT_CHUNK_WORDS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[Chunk]:
    """Splits `text` into flat windows of `words` words sharing `overlap` words with the previous one (see the module
    docs). Always at least one chunk."""
    o = resolve_chunk_options(ChunkOptions(words, overlap))
    spans = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    n = len(spans)
    if n == 0:
        return [Chunk(0, "", 0, 0, 0)]
    stride = o.words - o.overlap
    out: list[Chunk] = []
    first = 0
    while True:
        last = min(first + o.words, n)
        start, end = spans[first][0], spans[last - 1][1]
        out.append(Chunk(len(out), text[start:end], start, end, last - first))
        if last >= n:
            return out
        first += stride


def chunk_key(key: str, index: int) -> str:
    """The key chunk `index` of the document keyed `key` is stored under."""
    return f"{key}{CHUNK_KEY_SEPARATOR}{index}"


def chunk_index_of(key: str, parent: str) -> int | None:
    """The chunk index when `key` is `chunk_key(parent, i)` for some `i`, else None."""
    head = parent + CHUNK_KEY_SEPARATOR
    if not key.startswith(head):
        return None
    tail = key[len(head):]
    return int(tail) if _INDEX.fullmatch(tail) else None
