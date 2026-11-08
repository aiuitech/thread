import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./src/main.ts" with { type: "cf-worker" };

// One deployment: query serving + the build orchestrator. Build stages and writes run in dynamic Workers
// (LOADER, one isolate per slot id; see src/main.ts), which replaced the vib-bench-0..7 pool. AI (Workers AI) and
// EMBED_API_KEY (an OpenAI-compatible key; "-" = none) let the orchestrator embed texts for generations that name
// their embedding model (`manifest.embedding`): /query {text}, /upsert {texts}, /embed.
export default defineConfig({
	worker: {
		name: process.env.THREAD_WORKER_NAME ?? "vib-bench",
		compatibilityDate: "2026-10-01",
		entrypoint,
		// Build stages are CPU heavy; the Paid plan allows up to 5 minutes (slots set the same limit explicitly).
		limits: { cpuMs: 300_000 },
		observability: { enabled: true },
		env: {
			BUCKET: bindings.r2({ name: "vib-bench" }),
			BUILD_TOKEN: bindings.secret(),
			EMBED_API_KEY: bindings.secret(),
			LOADER: bindings.workerLoader(),
			AI: bindings.ai(),
			// Query traces (docs/observability.md): the share of /query requests whose trace envelope is logged as one
			// JSON line and written to QUERY_METRICS as one Analytics Engine data point. "0" = off.
			TRACE_SAMPLE: bindings.text("0"),
			QUERY_METRICS: bindings.analyticsEngineDataset({ name: "thread_queries" }),
		},
	},
});
