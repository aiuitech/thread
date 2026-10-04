#!/usr/bin/env bash
# Cold-start runs of the routing / base-pack modes (docs/query-path.md "Cold opens"): the TS SDK against fake_s3.mjs
# serving data/cold/base-overlay (the objects make_routing.mjs and make_base_pack.mjs wrote for g4 rev 180) over the
# proxy's R2 cache. Start the proxy first, on a port of its own:
#   node bench/cold/fake_s3.mjs --port 9761 --overlay data/cold/base-overlay --log data/cold/base-proxy.jsonl
#   PORT=9761 TAG=base N=10 PROFILE="25 200 0" bench/cold/run_base.sh
# PROFILE = "rtt-ms mbps link-mbps" (set on the proxy with /__config for the run); WARM=1 first runs every mode once
# at full speed so the proxy's cache holds every slab the measured runs touch. Outputs data/cold/$TAG-*.jsonl/.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT=${PORT:-9761}
TAG=${TAG:-base}
N=${N:-10}
WARM=${WARM:-1}
read -r RTT MBPS LINK <<<"${PROFILE:-25 200 0}"
PROXY="http://127.0.0.1:$PORT"
MODES=${MODES:-"f32-seq f32 f16 i8-seq i8 cold8 cold16 cold32 cold16-seq"}
args_of() {
	case $1 in
	f32-seq) echo "--routing f32 --base-pack 0" ;;
	f32) echo "--routing f32" ;;
	f16) echo "--routing f16" ;;
	i8-seq) echo "--routing i8 --base-pack 0" ;;
	i8) echo "--routing i8" ;;
	cold8) echo "--routing i8 --cold-routing tree --cold-beam 8" ;;
	cold16) echo "--routing i8 --cold-routing tree --cold-beam 16" ;;
	cold32) echo "--routing i8 --cold-routing tree --cold-beam 32" ;;
	cold16-seq) echo "--routing i8 --cold-routing tree --cold-beam 16 --base-pack 0" ;;
	*) echo "unknown mode $1" >&2; exit 1 ;;
	esac
}
config() { curl -sf -X POST "$PROXY/__config" -d "{\"rttMs\":$1,\"mbps\":$2,\"jitterMs\":0,\"connMs\":$((2 * $1)),\"linkMbps\":$3}" >/dev/null; }
if [ "$WARM" = 1 ]; then
	config 0 0 0
	for m in $MODES; do
		# shellcheck disable=SC2046
		node bench/cold/cold_bench.mjs --store proxy --proxy "$PROXY" --n "$N" $(args_of "$m") >/dev/null
	done
fi
config "$RTT" "$MBPS" "$LINK"
for m in $MODES; do
	out="data/cold/$TAG-ts-$m-rtt$RTT-$MBPS${LINK:+-link$LINK}"
	# shellcheck disable=SC2046
	node bench/cold/cold_bench.mjs --store proxy --proxy "$PROXY" --n "$N" --label "$TAG $m rtt$RTT/$MBPS/$LINK" --out "$out.jsonl" $(args_of "$m") | tee "$out.txt"
done
