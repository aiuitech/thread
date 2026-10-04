#!/usr/bin/env bash
# Copies the wasm-bindgen output of crates/vib-wasm (worker/pkg) into the SDK.
# Run after rebuilding the wasm:
#   cargo build --release --target wasm32-unknown-unknown -p vib-wasm
#   wasm-bindgen --target web --experimental-reset-state-function --out-dir worker/pkg --out-name vib \
#       target/wasm32-unknown-unknown/release/vib_wasm.wasm
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
src="${1:-$here/../worker/pkg}"
dst="$here/src/wasm"
mkdir -p "$dst"
for f in vib.js vib.d.ts vib_bg.wasm vib_bg.wasm.d.ts; do
  cp "$src/$f" "$dst/$f"
done
# Type of `import wasmModule from "vib-sdk/wasm"` (the bundler turns the .wasm into a WebAssembly.Module).
cat > "$dst/module.d.ts" <<'TS'
declare const wasmModule: WebAssembly.Module;
export default wasmModule;
TS
grep -q '__wbg_reset_state' "$dst/vib.js" || { echo "vib.js lacks __wbg_reset_state: build wasm-bindgen with --experimental-reset-state-function" >&2; exit 1; }
echo "synced $(wc -c < "$dst/vib_bg.wasm") bytes of wasm from $src into $dst"
