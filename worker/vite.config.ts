import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { cloudflare } from "@cloudflare/vite-plugin";
import { rolldown } from "rolldown";
import { defineConfig, type Plugin } from "vite";

const here = path.dirname(fileURLToPath(import.meta.url));
const STAGE_ENTRY = path.join(here, "src/index.ts");
const WASM_FILE = path.join(here, "pkg/vib_bg.wasm");
/** Module path of the wasm file inside a dynamic slot Worker (the orchestrator passes its own compiled module). */
const SLOT_WASM_PATH = "vib_bg.wasm";
const VIRTUAL_ID = "virtual:vib-stage-bundle";

/**
 * `import stageBundle from "virtual:vib-stage-bundle"` = {source, hash, wasmPath, bytes}: the stage code the build
 * slots run, i.e. a second bundle of the SAME entry (src/index.ts) that the Worker itself imports for query serving
 * and `?slot=local` execution, built with rolldown at `cf build` / `cf deploy` / `cf dev` time. Its wasm import is
 * left external as `./vib_bg.wasm`; the orchestrator (src/main.ts) passes the Worker's own compiled module under that
 * path, so the wasm binary ships once. `hash` (sha256 of the bundle JS and the wasm bytes) goes into the dynamic
 * Worker ids, so a deploy with changed code never reuses an isolate loaded with the old code.
 */
function stageBundle(): Plugin {
	return {
		name: "vib-stage-bundle",
		resolveId(id) {
			if (id === VIRTUAL_ID) return "\0" + VIRTUAL_ID;
		},
		async load(id) {
			if (id !== "\0" + VIRTUAL_ID) return;
			const bundle = await rolldown({
				input: STAGE_ENTRY,
				platform: "neutral",
				treeshake: true,
				plugins: [
					{
						name: "vib-slot-externals",
						resolveId(source) {
							if (source.endsWith("vib_bg.wasm")) return { id: `./${SLOT_WASM_PATH}`, external: true };
							if (source.startsWith("cloudflare:")) return { id: source, external: true };
						},
					},
				],
			});
			const { output } = await bundle.generate({ format: "esm", codeSplitting: false });
			await bundle.close();
			const chunks = output.filter((o) => o.type === "chunk");
			if (chunks.length !== 1) throw new Error(`stage bundle: expected 1 chunk, got ${chunks.length}`);
			const source = chunks[0].code;
			// Rebuild when the stage code or the wasm changes (cf dev).
			for (const f of chunks[0].moduleIds) if (fs.existsSync(f)) this.addWatchFile(f);
			this.addWatchFile(WASM_FILE);
			const hash = createHash("sha256").update(source).update(fs.readFileSync(WASM_FILE)).digest("hex").slice(0, 16);
			return `export default ${JSON.stringify({ source, hash, wasmPath: SLOT_WASM_PATH, bytes: source.length })};`;
		},
	};
}

export default defineConfig({
	plugins: [stageBundle(), cloudflare()],
});
