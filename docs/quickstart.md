# Quickstart

Build a search index and query it in about five minutes. You start with keyword search on your own machine, with
no account and no API key. Then you add semantic search, and then you move the same code to a Cloudflare Worker.

Choose your language. The two SDKs read and write the same index, so you can change later.

- [TypeScript (Node)](#typescript)
- [Python](#python)
- [Cloudflare Worker](#in-a-cloudflare-worker)

## TypeScript

### 1. Install

```sh
mkdir thread-demo && cd thread-demo
npm init -y && npm pkg set type=module
npm install @aiuitech/thread-sdk
```

### 2. Create an index, add documents, search

Save this as `demo.mjs`:

```js
import { readFileSync } from "node:fs";
import { Thread, createLexicalGeneration } from "@aiuitech/thread-sdk";
import { FileBlobStore } from "@aiuitech/thread-sdk/node";

// Load the engine (wasm). Do this once per process.
const wasmModule = new WebAssembly.Module(
  readFileSync("node_modules/@aiuitech/thread-sdk/dist/wasm/thread_bg.wasm"),
);
const thread = await Thread.init({ wasmModule });

// The index is a folder of files. In production this is a bucket.
const store = new FileBlobStore("./my-index");

// Create a text index. Do this once.
await createLexicalGeneration({ store, gen: "notes", lex: { text: true } });

// Three documents. Each has a key (its name), a text, and a meta string that comes back with a hit.
const keys = ["notes/deploy.md", "notes/billing.md", "notes/oncall.md"];
const texts = [
  "The deploy failed because the migration timed out. Roll back, then run it again.",
  "Invoices go out on the first of the month. Refunds take five days.",
  "When a deployment fails at night, page the on-call engineer.",
];
const metas = ["Deploy runbook", "Billing FAQ", "On-call guide"];

const index = await thread.open({ store });
await index.upsert({ keys, texts, metas });

// Search.
const { hits } = await index.search({ text: "deploy failed", topK: 3, docs: "text" });
for (const hit of hits) console.log(hit.score.toFixed(3), hit.doc.meta, "-", hit.doc.text);
```

```sh
node demo.mjs
```

```text
0.464 On-call guide - When a deployment fails at night, page the on-call engineer.
0.402 Deploy runbook - The deploy failed because the migration timed out. Roll back, then run it again.
```

An index is created only once. If you run the script again, the create call fails with `409: generation notes
already exists`. Delete `./my-index` to start again.

Two things to see here:

- The query "deploy failed" found "deployment fails". The text index reduces words to their stem, so different
  forms of a word match.
- Look in `./my-index`. That folder is the whole index. Copy it to a bucket and it works from there.

### 3. Add semantic search

Keyword search needs the right words. Semantic search finds text with the same meaning, through an embedding
model. thread does not run a model. You tell it which model the index is for, and how to call it.

This example uses OpenAI. Any OpenAI-compatible endpoint works, and so does
[Workers AI or your own function](embedding.md). Add these lines to the end of `demo.mjs`:

```js
const embedding = { provider: "openai", model: "text-embedding-3-small", dim: 1536 };

// Create an index for that model. `lex` adds the text index too, so searches are hybrid.
await thread.createGeneration({ store, gen: "notes-v2", embedding, lex: { text: true } });

// Open it and say how to call the model.
const semantic = await thread.open({ store, embed: { apiKey: process.env.OPENAI_API_KEY } });

// The same upsert as before. The index embeds each text for you.
await semantic.upsert({ keys, texts, metas });

// The same search as before. The query is embedded, BM25 runs too, and the two lists are merged.
const result = await semantic.search({ text: "what do I do when a release breaks?", topK: 3, docs: true });
console.log(result.mode);   // "hybrid"
for (const hit of result.hits) console.log(hit.score.toFixed(3), hit.doc.meta);
```

`createGeneration` makes the new index the current one, so `thread.open({ store })` now opens `notes-v2`.

If you already have vectors, pass them: `upsert({ keys, vectors, texts, metas })` and
`search({ vector, text })`.

## Python

### 1. Install

```sh
pip install thread-sdk
```

### 2. Create an index, add documents, search

```python
from thread_sdk import FileBlobStore, create_lexical_generation, open_index

# The index is a folder of files. In production this is a bucket.
store = FileBlobStore("./my-index")

# Create a text index. Do this once.
create_lexical_generation(store, "notes", lex={"text": True})

# Three documents. Each has a key (its name), a text, and a meta string that comes back with a hit.
keys = ["notes/deploy.md", "notes/billing.md", "notes/oncall.md"]
texts = [
    "The deploy failed because the migration timed out. Roll back, then run it again.",
    "Invoices go out on the first of the month. Refunds take five days.",
    "When a deployment fails at night, page the on-call engineer.",
]
metas = ["Deploy runbook", "Billing FAQ", "On-call guide"]

with open_index(store) as index:
    index.upsert(keys=keys, texts=texts, metas=metas)

    result = index.search(text="deploy failed", top_k=3, docs="text")
    for hit in result.hits:
        print(round(hit.score, 3), hit.doc.meta, "-", hit.doc.text)
```

```text
0.464 On-call guide - When a deployment fails at night, page the on-call engineer.
0.402 Deploy runbook - The deploy failed because the migration timed out. Roll back, then run it again.
```

### 3. Add semantic search

Add these lines to the end of the same script:

```python
import os
from thread_sdk import create_generation

embedding = {"provider": "openai", "model": "text-embedding-3-small", "dim": 1536}
create_generation(store, "notes-v2", embedding=embedding, lex={"text": True})

with open_index(store, embed={"api_key": os.environ["OPENAI_API_KEY"]}) as index:
    index.upsert(keys=keys, texts=texts, metas=metas)             # embedded for you
    result = index.search(text="what do I do when a release breaks?", top_k=3, docs=True)
    print(result.mode)   # "hybrid"
```

To use a model that you run yourself, pass a function: `open_index(store, embed=lambda texts, kind:
model.encode(texts))`.

## In a Cloudflare Worker

In a Worker, the store is an R2 bucket and the model can be Workers AI. The index code does not change.

You need two bindings: an R2 bucket (`BUCKET`) and Workers AI (`AI`).

```ts
import wasmModule from "@aiuitech/thread-sdk/wasm/thread_bg.wasm";
import { R2BlobStore, Thread, type ThreadIndex } from "@aiuitech/thread-sdk";

interface Env {
  BUCKET: R2Bucket;
  AI: Ai;
}

const embedding = {
  provider: "workers-ai",
  model: "@cf/baai/bge-base-en-v1.5",
  dim: 768,
  options: { pooling: "cls" },
} as const;

// Keep one index per isolate. Opening it loads a routing table, so do not open it on every request.
let index: Promise<ThreadIndex> | undefined;

function getIndex(env: Env, ctx: ExecutionContext) {
  if (!index) {
    index = Thread.init({ wasmModule }).then((thread) =>
      thread.open({ bucket: env.BUCKET, embed: { ai: env.AI }, waitUntil: (p) => ctx.waitUntil(p) }),
    );
    index.catch(() => (index = undefined));   // a failed open is tried again on the next request
  }
  return index;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    // Run once, to create the index in the bucket: curl -X POST https://<your-worker>/setup
    // It makes the new, empty index the live one. Never leave this route open.
    if (url.pathname === "/setup") {
      if (request.method !== "POST") return new Response("use POST", { status: 405 });
      const thread = await Thread.init({ wasmModule });
      await thread.createGeneration({ store: new R2BlobStore(env.BUCKET), gen: "docs", embedding, lex: { text: true } });
      return new Response("created");
    }

    const idx = await getIndex(env, ctx);

    if (url.pathname === "/add" && request.method === "POST") {
      const { key, text, title } = await request.json<{ key: string; text: string; title: string }>();
      // `chunk: true` splits a long text into passages of 200 words, each with its own vector.
      const result = await idx.upsert({ keys: [key], texts: [text], metas: [title], chunk: true });
      return Response.json({ rev: result.rev, chunks: result.chunks });
    }

    const q = url.searchParams.get("q") ?? "";
    const { hits } = await idx.search({ text: q, topK: 10, docs: true });
    return Response.json(hits.map((hit) => ({ score: hit.score, title: hit.doc?.meta })));
  },
};
```

Notes for Workers:

- **The wasm import.** Import the file by a path that ends in `.wasm`, as above. The Cloudflare Vite plugin and
  Wrangler both turn it into a `WebAssembly.Module`. For TypeScript, declare the module type once
  ([how](../sdk/ts/README.md#importing-the-wasm)).
- **The cache.** `open` puts the Workers Cache API in front of the bucket. Index files never change, so they stay
  cached. Pass `waitUntil` so that cache writes can finish after the response.
- **Protect the write routes.** This example has no authentication. `/setup` and `/add` change the index, so put
  your own authentication in front of them before you deploy, or create the index from a script and leave
  `/setup` out.

## What to read next

- [How it works](how-it-works.md): the ideas behind the index, in ten minutes.
- [Create an index](create-an-index.md): which kind of index to create, and how to load a lot of data.
- [Search](search.md) and [Write documents](writing.md): the two things you do every day.
- [Run it in production](production.md): batching, compaction, caching and the limits to know.
