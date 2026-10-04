#!/usr/bin/env bash
# Pack opens before / after the base-first layout (docs/cold-start.md "Routing base and base packs"): both SDKs,
# --store pack against the proxy of run_base.sh, the old pack (packs/g4-cold.threadpack: rev 179, key order) and
# the new one (packs/g4-base.threadpack: rev 180, base objects first + meta.base, built by
#   node bench/cold/make_pack.mjs --overlay data/cold/base-overlay --out data/cold/base-overlay/packs/g4-base.threadpack).
#   PORT=9761 TAG=base N=10 PROFILE="25 200 0" bench/cold/run_base_pack.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PORT=${PORT:-9761}
TAG=${TAG:-base}
N=${N:-10}
PY=${PY:-sdk/python/.venv/bin/python}
read -r RTT MBPS LINK <<<"${PROFILE:-25 200 0}"
PROXY="http://127.0.0.1:$PORT"
curl -sf -X POST "$PROXY/__config" -d "{\"rttMs\":$RTT,\"mbps\":$MBPS,\"jitterMs\":0,\"connMs\":$((2 * RTT)),\"linkMbps\":$LINK}" >/dev/null
run() { # sdk name pack-key args...
	local sdk=$1 name=$2 key=$3; shift 3
	local out="data/cold/$TAG-$sdk-pack-$name-rtt$RTT-$MBPS-link$LINK"
	if [ "$sdk" = ts ]; then node bench/cold/cold_bench.mjs --store pack --proxy "$PROXY" --pack-key "$key" --n "$N" --label "$TAG pack $name" --out "$out.jsonl" "$@" | tee "$out.txt"
	else "$PY" bench/cold/cold_bench.py --store pack --proxy "$PROXY" --pack-key "$key" --n "$N" --label "$TAG pack $name" --out "$out.jsonl" "$@" | tee "$out.txt"; fi
}
for sdk in ${SDKS:-ts py}; do
	run "$sdk" old packs/g4-cold.threadpack
	run "$sdk" new-f32 packs/g4-base.threadpack
	run "$sdk" new-i8 packs/g4-base.threadpack --routing i8
	run "$sdk" new-cold16 packs/g4-base.threadpack --routing i8 --cold-routing tree --cold-beam 16
done
