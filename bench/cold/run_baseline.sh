#!/usr/bin/env bash
# Reproduces the cold-start baseline of docs/cold-start.md (and any later comparison run with another TAG):
#   1. warm the proxy's disk cache for queries 0..29 (rtt 0) and build the sparse pack from it;
#   2. for each network profile (rtt25: 25 ms / 200 Mbps per response; rtt60: 60 ms / 100 Mbps; rtt25link500: rtt25
#      with all responses sharing one 500 Mbps link; conn +2 rtt everywhere), run both SDKs' benches against the
#      proxy (`--store proxy`) and the pack (`--store pack`);
#   3. both SDKs against the real R2 S3 endpoint (`--store r2s3`; R2S3=0 skips it).
# Outputs: data/cold/$TAG-{ts,py}-{proxy,pack}-{rtt25,rtt60}.jsonl, data/cold/$TAG-{ts,py}-r2s3.jsonl (+ .txt with
# the printed tables), proxy request logs data/cold/$TAG-proxy-*.jsonl. Compare runs with bench/cold/summarize.mjs.
#
#   bench/cold/run_baseline.sh                       # TAG=baseline N=10
#   TAG=after-prefetch N=10 R2S3=0 bench/cold/run_baseline.sh
#   WARM=0 PROFILES="rtt25link500" R2S3=0 SDKS=ts bench/cold/run_baseline.sh   # a subset, cache already warm
#   TAG=final-i8 WARM=0 R2S3=0 OVERLAY=data/cold/final-overlay PACK_KEY=packs/g4-base.threadpack ARGS="--routing i8" \
#     bench/cold/run_baseline.sh                   # another overlay / pack, extra cold_bench flags on every run
# OVERLAY (default data/cold/overlay), PACK_KEY (default packs/g4-cold.threadpack) and ARGS (extra cold_bench flags,
# proxy and pack runs only: the new objects exist only in the overlay) pick what the proxy serves and how the
# benches open it; R2S3_ARGS are the flags of the r2s3 runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${TAG:-baseline}
N=${N:-10}
R2S3=${R2S3:-1}
WARM=${WARM:-1}
SDKS=${SDKS:-ts py}
PROFILES=${PROFILES:-rtt25 rtt60 rtt25link500}
PORT=${PORT:-9555}
PY=${PY:-sdk/python/.venv/bin/python}
OVERLAY=${OVERLAY:-data/cold/overlay}
PACK_KEY=${PACK_KEY:-packs/g4-cold.threadpack}
ARGS=${ARGS:-}
R2S3_ARGS=${R2S3_ARGS:-}
PROXY_PID=
stop_proxy() { if [ -n "$PROXY_PID" ]; then kill "$PROXY_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; PROXY_PID=; fi; }
trap stop_proxy EXIT
start_proxy() { # name, proxy args...
	local name=$1; shift
	node bench/cold/fake_s3.mjs --port "$PORT" --overlay "$OVERLAY" --log "data/cold/$TAG-proxy-$name.jsonl" "$@" &
	PROXY_PID=$!
	for _ in $(seq 50); do curl -sf "http://127.0.0.1:$PORT/__config" >/dev/null && return; sleep 0.1; done
	echo "proxy did not start" >&2; exit 1
}
bench() { # sdk store profile [args...]
	local sdk=$1 store=$2 profile=$3; shift 3
	local out="data/cold/$TAG-$sdk-$store${profile:+-$profile}"
	# shellcheck disable=SC2086
	if [ "$store" = r2s3 ]; then set -- "$@" $R2S3_ARGS; else set -- --proxy "http://127.0.0.1:$PORT" --pack-key "$PACK_KEY" "$@" $ARGS; fi
	if [ "$sdk" = ts ]; then node bench/cold/cold_bench.mjs --store "$store" --n "$N" --label "$TAG $profile" --out "$out.jsonl" "$@" | tee "$out.txt"
	else "$PY" bench/cold/cold_bench.py --store "$store" --n "$N" --label "$TAG $profile" --out "$out.jsonl" "$@" | tee "$out.txt"; fi
}

profile_args() {
	case $1 in
	rtt25) echo "--rtt-ms 25 --mbps 200" ;;
	rtt60) echo "--rtt-ms 60 --mbps 100" ;;
	rtt25link500) echo "--rtt-ms 25 --mbps 200 --link-mbps 500" ;;
	*) echo "unknown profile $1" >&2; exit 1 ;;
	esac
}

mkdir -p data/cold
if [ "$WARM" = 1 ]; then
start_proxy warm --rtt-ms 0 --mbps 0 --conn-ms 0
node bench/cold/warm_proxy.mjs --proxy "http://127.0.0.1:$PORT" --from 0 --count $((N * 3)) --keys-out data/cold/warm-keys.json
node bench/cold/warm_proxy.mjs --proxy "http://127.0.0.1:$PORT" --from 0 --count $((N * 3))   # must say: upstream fetches 0
node bench/cold/make_pack.mjs --proxy "http://127.0.0.1:$PORT" | tee data/cold/pack-info.json
stop_proxy
fi

for name in $PROFILES; do
	start_proxy "$name" $(profile_args "$name")
	for store in proxy pack; do
		for sdk in $SDKS; do bench "$sdk" "$store" "$name"; done
	done
	stop_proxy
done

if [ "$R2S3" = 1 ]; then
	for sdk in $SDKS; do bench "$sdk" r2s3 ""; done
fi
