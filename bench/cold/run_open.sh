#!/usr/bin/env bash
# The cold/open experiments (warm packs, tree-ordered packs with cross-slab coalescing, an inline manifest in
# current.json) on the baseline's harness: the same benches, query set and network profiles, but an OFFLINE proxy
# (never contacts R2; the objects the bench reads are already in data/cold/objects) on its own port with its own
# overlay (data/cold/open-overlay: the new objects, plus a link to the baseline pack), so it can run next to other
# experiments sharing data/cold.
#
#   TAG=open-warm256 ARGS="--warm-pack warm/g4-hot256.threadpack" bench/cold/run_open.sh
#   TAG=open-tree STORES=pack SDKS=ts ARGS="--pack-key packs/g4-cold-tree.threadpack --coalesce-across-slabs 65536" bench/cold/run_open.sh
#   OVERLAY=data/cold/open-overlay-inline TAG=open-inline bench/cold/run_open.sh
#
# Outputs data/cold/open/$TAG-{ts,py}-{proxy,pack}-{profile}.jsonl (+ .txt). Compare with summarize.mjs.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${TAG:?set TAG}
N=${N:-10}
SDKS=${SDKS:-ts py}
STORES=${STORES:-proxy}
PROFILES=${PROFILES:-rtt25}
PORT=${PORT:-9575}
OVERLAY=${OVERLAY:-data/cold/open-overlay}
ARGS=${ARGS:-}
PY=${PY:-sdk/python/.venv/bin/python}
OUT=data/cold/open
PROXY_PID=
stop_proxy() { if [ -n "$PROXY_PID" ]; then kill "$PROXY_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; PROXY_PID=; fi; }
trap stop_proxy EXIT
profile_args() {
	case $1 in
	rtt25) echo "--rtt-ms 25 --mbps 200" ;;
	rtt60) echo "--rtt-ms 60 --mbps 100" ;;
	rtt25link500) echo "--rtt-ms 25 --mbps 200 --link-mbps 500" ;;
	*) echo "unknown profile $1" >&2; exit 1 ;;
	esac
}
mkdir -p "$OUT"
for profile in $PROFILES; do
	node bench/cold/fake_s3.mjs --port "$PORT" --offline --overlay "$OVERLAY" $(profile_args "$profile") > "$OUT/proxy-$TAG-$profile.out" 2>&1 &
	PROXY_PID=$!
	for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/__config" >/dev/null && break; sleep 0.1; done
	for store in $STORES; do
		for sdk in $SDKS; do
			out="$OUT/$TAG-$sdk-$store-$profile"
			# shellcheck disable=SC2086
			if [ "$sdk" = ts ]; then node bench/cold/cold_bench.mjs --store "$store" --proxy "http://127.0.0.1:$PORT" --n "$N" --label "$TAG $profile" --out "$out.jsonl" $ARGS | tee "$out.txt"
			else "$PY" bench/cold/cold_bench.py --store "$store" --proxy "http://127.0.0.1:$PORT" --n "$N" --label "$TAG $profile" --out "$out.jsonl" $ARGS | tee "$out.txt"; fi
		done
	done
	stop_proxy
done
