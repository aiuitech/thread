declare module "*.wasm" {
	const module: WebAssembly.Module;
	export default module;
}

/** The stage bundle the build slots run (vite.config.ts, plugin thread-stage-bundle). */
declare module "virtual:thread-stage-bundle" {
	const bundle: { source: string; hash: string; wasmPath: string; bytes: number };
	export default bundle;
}
