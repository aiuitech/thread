# Changelog

Versions are shared by the npm package `@aiuitech/thread-sdk`, the PyPI package `thread-sdk` and the Rust crates
(`Cargo.toml` workspace version). A release is a `v<version>` tag; `.github/workflows/release.yml` builds the wasm
from source, runs every check and publishes.

## 0.2.0 (2026-11-03)

First published version.

- **Embedding models** ([docs/embedding.md](docs/embedding.md)): `manifest.embedding` names the model a dense
  generation is for; `open({ embed })` / `open_index(embed=...)` takes the Workers AI binding, Workers AI REST
  credentials, an OpenAI-compatible key or a function; `search({ text })` embeds the query (and still fuses with
  BM25), `upsert({ texts })` embeds documents; `index.embed(texts, kind)`, `index.canEmbed`, `index.embedding`.
- **A dense generation from nothing**: `createGeneration` / `create_generation` / the Worker's `POST /create`
  publish one empty cell for a model (or a bare `dim`); it grows through upserts, the Worker's automatic compaction
  and cell splits. Readers clamp a manifest default `nprobe` to the cell count.
- **Worker**: `AI` and `EMBED_API_KEY` bindings; JSON `/query {text}` and `/upsert {texts}` are embedded in the
  orchestrator before forwarding; `POST /embed`.
- The TypeScript package is now `@aiuitech/thread-sdk` (was the unpublished `thread-sdk`); the import paths
  `@aiuitech/thread-sdk`, `@aiuitech/thread-sdk/node` and `@aiuitech/thread-sdk/wasm/thread_bg.wasm`.

Before 0.2.0 (unreleased, see the git history): blob-native dense index in R2/S3 (two-tier slabs, PQ scan tier,
centroid tree routing), BM25 segments and fusion in thread-core, keys and the docs store, attribute filters,
chunking, incremental updates with compaction and cell splitting, packs, cold-start work, the Python SDK.
