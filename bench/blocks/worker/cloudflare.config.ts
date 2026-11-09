import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./src/main.ts" with { type: "cf-worker" };

// The block-cache bench (bench/blocks/README.md): read-only searches over indexes under `tmp/blockbench/` of the
// vib-bench bucket, plus uploads and deletes under that prefix only. BENCH_NAME names the deployment (one per SDK
// build: THREAD_SDK points at the SDK's sdk/ts directory, default this checkout's).
export default defineConfig({
	worker: {
		name: process.env.BENCH_NAME ?? "thread-blockbench",
		compatibilityDate: "2026-10-01",
		entrypoint,
		observability: { enabled: true },
		env: {
			BUCKET: bindings.r2({ name: "vib-bench" }),
			BENCH_TOKEN: bindings.secret(),
			SDK_LABEL: bindings.text(process.env.SDK_LABEL ?? "checkout"),
		},
	},
});
