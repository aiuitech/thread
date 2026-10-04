import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./src/index.ts" with { type: "cf-worker" };

/**
 * vib-sdk example: POST /search over the `vib-bench` index, through the R2 binding by default or through R2's
 * S3 API (`S3BlobStore`) when `?store=s3` / `{"store":"s3"}` is given and the S3_* secrets are set.
 * Secrets (deploy with `--secrets-file .env.secrets`): AUTH_TOKEN (bearer token for every endpoint but /health;
 * "-" disables auth), S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY ("-" = not configured).
 */
export default defineConfig({
	worker: {
		name: "vib-sdk-example",
		compatibilityDate: "2026-10-01",
		entrypoint,
		observability: { enabled: true },
		env: {
			BUCKET: bindings.r2({ name: "vib-bench" }),
			AUTH_TOKEN: bindings.secret(),
			S3_ENDPOINT: bindings.secret(),
			S3_BUCKET: bindings.secret(),
			S3_ACCESS_KEY_ID: bindings.secret(),
			S3_SECRET_ACCESS_KEY: bindings.secret(),
		},
	},
});
