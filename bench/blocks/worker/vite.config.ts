import path from "node:path";
import { cloudflare } from "@cloudflare/vite-plugin";
import { defineConfig } from "vite";

// The SDK is bundled from source, so one Worker can be built against any checkout (THREAD_SDK = its sdk/ts).
const sdk = path.resolve(process.env.THREAD_SDK ?? path.join(import.meta.dirname, "../../../sdk/ts"));

export default defineConfig({
	plugins: [cloudflare()],
	resolve: {
		alias: [
			{ find: "thread-sdk/wasm", replacement: path.join(sdk, "src/wasm/thread_bg.wasm") },
			{ find: /^thread-sdk$/, replacement: path.join(sdk, "src/index.ts") },
		],
	},
});
