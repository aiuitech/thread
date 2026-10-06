"""thread-sdk: query and update a thread blob-native search index (dense vectors, BM25 text, or both fused by rank)
from Python. The index lives as immutable objects in R2, any S3-compatible store or a local directory; the numeric
kernels are the same Rust code (`thread-core`) the Worker and the TypeScript SDK run, as a native extension.

    from thread_sdk import S3BlobStore, open_index

    store = S3BlobStore.r2(account_id, "my-bucket", key_id, secret)      # or open_index("/data/idx"), a directory,
    with open_index(store) as index:                                      # or open_index("idx.threadpack"), one file
        r = index.search(text="deploy failed", vector=embedding, top_k=10, docs=True)
        hits = index.query(embedding, top_k=10).hits      # dense only
"""

from . import _native
from .attrs import compile_filter, define_attrs, filter_matches, pack_attrs, unpack_attrs, validate_schema
from .chunk import (CHUNK_KEY_SEPARATOR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_WORDS, Chunk, ChunkOptions, chunk_index_of,
                    chunk_key, chunk_text, resolve_chunk_options)
from .index import KeyRecord, Rank, SearchHit, SearchResult, ThreadIndex, head_revision, load_rev_state, open_index, resolve_target
from .lex import DocRecord, KeyPage, LexHit, LexIndex, LexSearchResult, LexSegment
from .pack import PackedBlobStore, PackResult, PackWriter, copy_objects, pack_index, pack_to_bytes, store_at
from .query import MULTI_COALESCE, QUERY_INFLIGHT, Hit, QueryResult, RevState, coalesce_runs
from .revision import snapshot_keys
from .s3 import S3_INFLIGHT, S3BlobStore, SigningKeyCache, sign_v4
from .store import (
    BlobCache,
    BlobHead,
    BlobRange,
    BlobStore,
    BlobTrace,
    CachedBlobStore,
    DiskCache,
    FileBlobStore,
    MemoryBlobStore,
    MISS_POLICIES,
    MemoryCache,
    PrefixedBlobStore,
)
from .warm import WarmPrefixStore, build_warm_pack, parse_warm_pack, seed_warm_pack, write_warm_pack
from .wire import ResolvedLexConfig, ThreadError, keys
from .write import (
    MAX_UPSERT,
    CompactSegmentsResult,
    ConfigureAttrsResult,
    ConfigureLexicalResult,
    DeleteResult,
    UpsertResult,
    create_lexical_generation,
)

MissingRanges = _native.MissingRanges
__version__ = _native.__version__

__all__ = [
    "open_index",
    "ThreadIndex",
    "ThreadError",
    "MissingRanges",
    "QueryResult",
    "Hit",
    "SearchResult",
    "SearchHit",
    "Rank",
    "LexSearchResult",
    "LexHit",
    "DocRecord",
    "KeyRecord",
    "KeyPage",
    "UpsertResult",
    "DeleteResult",
    "ConfigureLexicalResult",
    "ConfigureAttrsResult",
    "CompactSegmentsResult",
    "create_lexical_generation",
    "define_attrs",
    "validate_schema",
    "pack_attrs",
    "unpack_attrs",
    "compile_filter",
    "filter_matches",
    "MAX_UPSERT",
    "Chunk",
    "ChunkOptions",
    "chunk_text",
    "chunk_key",
    "chunk_index_of",
    "resolve_chunk_options",
    "DEFAULT_CHUNK_WORDS",
    "DEFAULT_CHUNK_OVERLAP",
    "CHUNK_KEY_SEPARATOR",
    "QUERY_INFLIGHT",
    "BlobStore",
    "BlobRange",
    "BlobHead",
    "BlobTrace",
    "MemoryBlobStore",
    "FileBlobStore",
    "S3BlobStore",
    "PrefixedBlobStore",
    "PackedBlobStore",
    "PackWriter",
    "PackResult",
    "pack_index",
    "pack_to_bytes",
    "copy_objects",
    "store_at",
    "WarmPrefixStore",
    "build_warm_pack",
    "write_warm_pack",
    "parse_warm_pack",
    "seed_warm_pack",
    "snapshot_keys",
    "CachedBlobStore",
    "BlobCache",
    "MemoryCache",
    "DiskCache",
    "ResolvedLexConfig",
    "RevState",
    "LexIndex",
    "LexSegment",
    "keys",
    "sign_v4",
    "SigningKeyCache",
    "S3_INFLIGHT",
    "MULTI_COALESCE",
    "MISS_POLICIES",
    "coalesce_runs",
    "resolve_target",
    "load_rev_state",
    "head_revision",
]
