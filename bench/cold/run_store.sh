#!/usr/bin/env bash
# The cold-path option matrix of docs/cold-start.md ("Store options"): every store / cache option of the SDKs,
# both SDKs, against the fake S3 proxy (two network profiles) and, with R2S3=1, the real R2 S3 endpoint.
#
#   bench/cold/run_store.sh                  # N=10, proxy on PORT=9655, profiles rtt25 + rtt60, then R2 S3
#   ONLY="ts-default py-default" R2S3=0 bench/cold/run_store.sh
#
# Needs the proxy cache warmed for queries 0..29 (bench/cold/run_baseline.sh does that) and sdk/ts/dist built.
# Outputs data/cold-store/{profile}-{sdk}-{config}.jsonl (+ .txt). Never writes to the bucket.
set -euo pipefail
cd "$(dirname "$0")/../.."
N=${N:-10}
PORT=${PORT:-9655}
R2S3=${R2S3:-1}
PROFILES=${PROFILES:-"rtt25 rtt60"}
PY=${PY:-sdk/python/.venv/bin/python}
OUT=${OUT:-data/cold-store}
mkdir -p "$OUT"

# name | args (both SDKs unless the name starts with ts- / py-)
CONFIGS=$(cat <<'EOF'
legacy|--legacy
default|
default-drain|--drain
whole-i12|--cache memory --miss-policy whole --inflight 12 --warmup false
range-i12|--cache memory --miss-policy range --inflight 12 --warmup false
rangewarm-i12|--cache memory --miss-policy range+warm --inflight 12 --warmup false
rangewarm-i12-drain|--cache memory --miss-policy range+warm --inflight 12 --warmup false --drain
whole-i32|--cache memory --miss-policy whole --warmup false
inflight6|--warmup false --inflight 6
inflight12|--warmup false --inflight 12
inflight32|--warmup false --inflight 32
coalesce32k|--warmup false --coalesce 32768
coalesce128k|--warmup false --coalesce 131072
coalesce512k|--warmup false --coalesce 524288
span256k|--warmup false --span 262144
span1m|--warmup false --span 1048576
warmup0|--warmup false
warmup8|--warmup 8
warmup32|--warmup 32
cold-np20-rr100|--cold-nprobe 20 --cold-rerank 100
cold-np24-rr150|--cold-nprobe 24
disk-fill|--cache disk --drain
disk-restart|--cache disk
EOF
)

want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }

run() { # sdk profile name args...
	local sdk=$1 profile=$2 name=$3; shift 3
	want "$sdk-$name" || return 0
	local out="$OUT/$profile-$sdk-$name"
	local store=proxy extra=(--proxy "http://127.0.0.1:$PORT")
	if [ "$profile" = r2s3 ]; then store=r2s3; extra=(); fi
	# disk-fill fills one DiskCache directory per trial (draining the background warms); disk-restart, a new process
	# per trial, then runs the same queries against it: a restart with a warm disk cache.
	if [[ $name == disk-* ]]; then set -- "$@" --disk-dir "$OUT/dc-$sdk-{t}"; fi
	if [[ $name == disk-fill ]]; then rm -rf "$OUT"/dc-"$sdk"-*; fi
	if [ "$sdk" = ts ]; then node bench/cold/cold_bench.mjs --store "$store" "${extra[@]}" --n "$N" --label "$profile $name" --out "$out.jsonl" "$@" > "$out.txt"
	else "$PY" bench/cold/cold_bench.py --store "$store" "${extra[@]}" --n "$N" --label "$profile $name" --out "$out.jsonl" "$@" > "$out.txt"; fi
	echo "$profile $sdk $name: $(grep -E '^ms query total' "$out.txt" | tr -s ' ')"
}

matrix() { # profile
	while IFS='|' read -r name args; do
		[ -n "$name" ] || continue
		# shellcheck disable=SC2086
		run ts "$1" "$name" $args
		# shellcheck disable=SC2086
		run py "$1" "$name" $args
	done <<< "$CONFIGS"
}

for profile in $PROFILES; do
	case $profile in
	rtt25) cfg='{"rttMs":25,"mbps":200,"jitterMs":0,"connMs":50,"linkMbps":0}' ;;
	rtt60) cfg='{"rttMs":60,"mbps":100,"jitterMs":0,"connMs":120,"linkMbps":0}' ;;
	*) echo "unknown profile $profile" >&2; exit 1 ;;
	esac
	curl -sf -X POST -H 'content-type: application/json' -d "$cfg" "http://127.0.0.1:$PORT/__config" > /dev/null
	matrix "$profile"
done
if [ "$R2S3" = 1 ]; then matrix r2s3; fi
