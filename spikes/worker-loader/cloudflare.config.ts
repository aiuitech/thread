import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./src/index.ts" with { type: "cf-worker" };

// Spike: an orchestrator Worker that runs build jobs in dynamically loaded Workers
// (Worker Loader binding), one dynamic Worker per slot, instead of the vib-bench-0..7 pool.
export default defineConfig({
	worker: {
		name: "vib-loader-spike",
		compatibilityDate: "2026-10-01",
		entrypoint,
		limits: { cpuMs: 300_000 },
		observability: { enabled: true },
		env: {
			BUCKET: bindings.r2({ name: "vib-bench" }),
			BUILD_TOKEN: bindings.secret(),
			LOADER: bindings.workerLoader(),
		},
	},
});
