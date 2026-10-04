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
from .fusion import DEFAULT_RRF_K, rrf
from .index import Rank, SearchHit, SearchResult, ThreadIndex, head_revision, load_rev_state, open_index, resolve_target
from .lex import DocRecord, LexHit, LexIndex, LexSearchResult, LexSegment
from .pack import PackedBlobStore, PackResult, PackWriter, copy_objects, pack_index, pack_to_bytes, store_at
from .query import QUERY_INFLIGHT, Hit, QueryResult, RevState
from .revision import snapshot_keys
from .s3 import S3BlobStore, sign_v4
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
    MemoryCache,
    PrefixedBlobStore,
)
from .wire import ResolvedLexConfig, ThreadError, keys
from .write import (
    MAX_UPSERT,
    CompactSegmentsResult,
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
    "UpsertResult",
    "DeleteResult",
    "ConfigureLexicalResult",
    "CompactSegmentsResult",
    "create_lexical_generation",
    "MAX_UPSERT",
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
    "rrf",
    "DEFAULT_RRF_K",
    "sign_v4",
    "resolve_target",
    "load_rev_state",
    "head_revision",
]
