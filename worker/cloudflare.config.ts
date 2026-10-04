import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./src/main.ts" with { type: "cf-worker" };

// One deployment: query serving + the build orchestrator. Build stages and writes run in dynamic Workers
// (LOADER, one isolate per slot id; see src/main.ts), which replaced the vib-bench-0..7 pool.
export default defineConfig({
	worker: {
		name: process.env.VIB_WORKER_NAME ?? "vib-bench",
		compatibilityDate: "2026-10-01",
		entrypoint,
		// Build stages are CPU heavy; the Paid plan allows up to 5 minutes (slots set the same limit explicitly).
		limits: { cpuMs: 300_000 },
		observability: { enabled: true },
		env: {
			BUCKET: bindings.r2({ name: "vib-bench" }),
			BUILD_TOKEN: bindings.secret(),
			LOADER: bindings.workerLoader(),
		},
	},
});
