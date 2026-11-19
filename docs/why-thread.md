# Why thread

Most search products are services: you send your data to a server, and the server answers your queries. thread is
different. It is a file format and a library. The index is files in storage that you own, and the library reads
them from wherever your code runs.

This page explains why that design is useful, where it is not, and how thread compares with the other tools you
might choose.

## The idea

Three decisions make thread what it is.

**1. The index is immutable files in object storage.**
Object storage is the cheapest and most durable storage there is, and every platform has it. An index that is
only files needs no server to hold it. Because the files never change, any cache can keep them for ever: an edge
cache, a disk, memory.

**2. A query reads small ranges, not whole files.**
A query on 1 million vectors reads about 6 MB, from a few dozen files, and does about 33 ms of arithmetic. So the
reader can be small: a Cloudflare Worker with 128 MB of memory, a Lambda, a laptop.

**3. The reader is a library.**
The engine is Rust. It compiles to wasm for Workers and Node, and to a native extension for Python. Your code
calls it directly. There is no network hop to a search service, and no second system to secure, monitor and pay
for.

## What you get from that

| Benefit | Why |
|---|---|
| **Low cost** | You pay for object storage and for your own compute. There is no charge per vector or per query, and no monthly minimum. An index nobody queries costs only its storage. |
| **Nothing to operate** | No cluster, no replicas, no version upgrades of a server. If your Worker runs, search runs. |
| **You own the data** | The index is in your bucket, in a documented format, under an MIT licence. You can copy it, back it up and leave. |
| **Portable** | The same index opens in a Worker, in Node, in Python, from a folder, or from one pack file with no network. Build it in one place and query it in another. |
| **Scales with your platform** | Readers hold no state, so 1 reader and 1,000 readers are the same. One Worker deployment served about 2,500 queries a second on the 1M benchmark. |
| **No half-written index** | A write becomes visible as one complete revision, or not at all. A reader that is pinned to a revision always gets the same answer. |
| **Hybrid search included** | Vectors and BM25 text share one index and one set of ids. You do not need a second system for keyword search. |

## When thread is a good choice

- You build on Cloudflare Workers and want search without a second service.
- You have many small or medium indexes (one per customer, per project, per device), and most are idle most of the
  time. Idle indexes cost only storage.
- Your index is read much more often than it is written.
- You want the same index on a server and on a device, or you want to ship an index as a file.
- Cost matters more than the last few milliseconds of latency.
- You must keep the data in storage that you control.

## When to choose something else

Be honest with yourself about these. thread is not the best tool for every job.

| You need | Why thread is a poor fit | Look at |
|---|---|---|
| The lowest latency for each query | A query is a few dozen cached reads. The median on the 1M benchmark is about 90 ms from a Worker. A managed service was 31 ms. | Vectorize, Pinecone, an in-memory engine |
| Rich filters on metadata | Filters use 32 bits of fields per document: a tenant, a language, a few flags, a coarse date. There is no `$or`, and no filter on strings or large numbers. | Postgres with pgvector, Elasticsearch, Qdrant |
| Many small writes, visible at once | Every write is a new revision. Writes are batched and become visible to other readers after about 30 seconds. | A database |
| Distance other than cosine | thread normalizes vectors and uses cosine similarity only. | Most vector databases |
| Transactions with your other data | The index is separate from your database. | pgvector |
| A vendor with a support contract and an SLA | thread is an open-source library. | A managed service |
| A stable, mature product today | thread is at version 0.2. The SDK write path is marked experimental. | A managed service |

[Limits and status](limits.md) has the full list.

## Compared with other tools

The facts about other products below come from their public documentation, as read in November 2026. Products
change. Check their current documentation before you decide.

### At a glance

| | thread | Cloudflare Vectorize | Pinecone | turbopuffer | Amazon S3 Vectors | pgvector | LanceDB |
|---|---|---|---|---|---|---|---|
| What it is | A library over files | A managed service | A managed service | A managed service | A managed service | A Postgres extension | A library over files, and a cloud service |
| Where the index is | Your bucket or disk | Cloudflare | Pinecone | turbopuffer's object storage | AWS | Your Postgres | Your disk or bucket |
| What you run | Nothing | Nothing | Nothing | Nothing | Nothing | A Postgres server | Nothing (library) |
| Runs inside a Cloudflare Worker | Yes | Yes, as a binding | Over HTTP | Over HTTP | Over HTTP | Over a connection | No |
| Keyword search | Yes (BM25) | No | Yes (sparse vectors) | Yes (BM25) | No | Postgres full-text | Yes |
| Works offline | Yes | No | No | No | No | With a local server | Yes |
| Price model | Storage and your compute | Per stored and queried dimension | Usage, with a monthly minimum on paid plans | Usage, with a monthly minimum | Storage, writes and queries | A server | Free library, or a cloud plan |
| Open source | Yes (MIT) | No | No | No | No | Yes | Yes |

### Cloudflare Vectorize

Vectorize is the closest alternative if you build on Workers. Both are called from a Worker, and thread's `query`,
`upsert` and `delete` map almost one to one onto Vectorize's.

This is the only comparison we measured ourselves. The dataset is dbpedia-openai-1M (975,000 vectors, 1536
dimensions, cosine, top 10). The Vectorize numbers are the ones Cloudflare published for the same dataset. thread
was one Worker that read R2 through a warm edge cache, on 2026-10-05.

| | Vectorize | thread |
|---|---:|---:|
| Recall@10 | 95.4% | 95.1% |
| Median latency | 31 ms | 88 to 90 ms |
| p95 latency | 380 ms | 176 to 193 ms |
| Throughput | 343 requests/s | 2,495 to 2,785 requests/s |
| Cost, 1M vectors and 1M queries a month | about $31 | about $0.10 |
| Cost, 1M vectors and 10M queries a month | about $169 | about $6 |

Read the table this way:

- **Recall is the same.** Both find about 95 of the 100 true nearest results.
- **Vectorize is faster for a typical query.** Its median is about 3 times lower.
- **thread is steadier under load.** Its p95 latency is half, and it serves 7 to 8 times the requests.
- **thread is much cheaper**, if the edge cache is warm. The costs assume it is. Each cache miss is one R2 read, so
  a cold workload costs more: up to about $2.60 at 1M queries if every query were cold.

Choose Vectorize when you want the lowest median latency, more than 32 bits of filterable metadata, euclidean or
dot-product distance, or a fully managed product. Choose thread when cost, hybrid search, ownership of the data or
portability matter more. [Migrate from Vectorize](migrate-from-vectorize.md) maps every call.
[Benchmarks](benchmarks.md) has the method and every run.

### Pinecone

Pinecone is a mature managed vector database with a large feature set: namespaces, rich metadata filters, sparse
and dense vectors, hosted embedding and reranking models. You call it over HTTP from anywhere.

Choose Pinecone when you want a complete managed product and your budget allows a monthly minimum on the paid
plans. Choose thread when you want the index in your own storage, when you run on Workers and do not want a
network hop to another provider for each query, or when you have many indexes that are mostly idle.

### turbopuffer

turbopuffer has the design nearest to thread's: the index is in object storage, and queries read it through
caches. It supports vectors, BM25 and filters. The difference is who runs it. turbopuffer is a service: its
servers hold the caches, do the writes and the indexing, and you pay for usage with a monthly minimum. thread is
a library: your Worker is the query node, your bucket is the storage, and nothing runs when nobody queries.

Choose turbopuffer when you want object-storage economics at a very large scale (hundreds of millions of
documents and more) with richer filters, and you are happy to use a service. Choose thread when you want no
service at all, or an index that also works offline.

### Amazon S3 Vectors

S3 Vectors stores and queries vectors inside S3, with no servers to manage. It is built for very large,
infrequently queried data: AWS documents sub-second latency for infrequent queries and about 100 ms for frequent
ones. It is vector search only, and it is an AWS API: the index is not files that you can read or move yourself.

Choose S3 Vectors when you are on AWS and need billions of vectors at low cost. Choose thread when you want text
and hybrid search, an index you can copy and open anywhere, or you run on Cloudflare.

### pgvector

pgvector adds vector search to Postgres. If your data is already in Postgres, it is the simplest choice: one
database, SQL filters and joins, and transactions that cover your rows and your vectors together.

Choose pgvector when your vectors belong beside relational data and you already run Postgres. Choose thread when
you do not want a database server that is always on, when the index is large compared with the database, or when
queries come from the edge and a database connection is a long way off.

### LanceDB

LanceDB is also a library over files, with an open columnar format. It runs embedded in Python, Node and Rust and
can keep its data in object storage. It is a strong choice for data science and for multimodal data.

The main differences: thread runs inside a Cloudflare Worker (a 128 MB wasm isolate), and its file layout is
designed for that: small range reads through an edge cache. LanceDB does not run in Workers. Choose LanceDB for
analytics-style work in Python or on servers. Choose thread for search served from the edge.

### Elasticsearch, OpenSearch, Typesense, Qdrant, Weaviate

These are search servers. They give you the most features: complex filters, facets, aggregations, many distance
functions, and years of tuning. They also need a cluster that is always on, with memory in proportion to the
index.

Choose one of these when search is the core of your product and you need those features. Choose thread when
search is one feature among many and you do not want to run a cluster for it.

## Next

- [Quickstart](quickstart.md): try it in five minutes.
- [How it works](how-it-works.md): the design, with pictures.
- [Benchmarks](benchmarks.md): every number, with the method.
