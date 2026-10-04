#!/usr/bin/env bash
# The store-option matrix of docs/cold-start.md ("Store-level optimizations"): every store / cache option of the SDKs,
# both SDKs, against the fake S3 proxy (two network profiles) and, with R2S3=1, the real R2 S3 endpoint.
#
#   bench/cold/run_store.sh                  # N=10, proxy on PORT=9655, profiles rtt25 rtt25link500 rtt60, then R2 S3
#   ONLY="ts-default py-default" R2S3=0 bench/cold/run_store.sh
#   NAMES="default coalesce128k" PROFILES=rtt25 R2S3=0 bench/cold/run_store.sh
#
# Needs the proxy cache warmed for queries 0..29 (bench/cold/run_baseline.sh does that) and sdk/ts/dist built.
# Outputs data/cold-store/{profile}-{sdk}-{config}.jsonl (+ .txt). Never writes to the bucket.
set -euo pipefail
cd "$(dirname "$0")/../.."
N=${N:-10}
PORT=${PORT:-9655}
R2S3=${R2S3:-1}
PROFILES=${PROFILES:-"rtt25 rtt25link500 rtt60"}
PY=${PY:-sdk/python/.venv/bin/python}
OUT=${OUT:-data/cold-store}
mkdir -p "$OUT"

# name | args. Every config is run for both SDKs.
CONFIGS=$(cat <<'EOF'
legacy|--legacy
default|
default-drain|--drain
whole-i32|--cache memory --miss-policy whole
range-i32|--cache memory --miss-policy range
inflight12|--inflight 12
coalesce32k|--coalesce 32768
coalesce128k|--coalesce 131072
coalesce512k|--coalesce 524288
span1m|--span 1048576
warmup0|--warmup false
early-warmup|--early-warmup
cold-np20-rr100|--cold-nprobe 20 --cold-rerank 100
disk-fill|--cache disk --drain
disk-restart|--cache disk
EOF
)
# configs per profile (NAMES overrides)
declare -A SET=(
	[rtt25]="legacy default default-drain whole-i32 range-i32 inflight12 coalesce32k coalesce128k coalesce512k span1m warmup0 early-warmup cold-np20-rr100 disk-fill disk-restart"
	[rtt25link500]="legacy default default-drain whole-i32 range-i32 coalesce32k coalesce128k coalesce512k span1m cold-np20-rr100"
	[rtt60]="legacy default whole-i32 coalesce128k coalesce512k cold-np20-rr100"
	[r2s3]="legacy default whole-i32 coalesce128k coalesce512k cold-np20-rr100 disk-fill disk-restart"
)

want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }
inset() { local names=${NAMES:-${SET[$1]}}; [[ " $names " == *" $2 "* ]]; }

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
		inset "$1" "$name" || continue
		# shellcheck disable=SC2086
		run ts "$1" "$name" $args
		# shellcheck disable=SC2086
		[ "$name" = early-warmup ] || run py "$1" "$name" $args
	done <<< "$CONFIGS"
}

for profile in $PROFILES; do
	case $profile in
	rtt25) cfg='{"rttMs":25,"mbps":200,"jitterMs":0,"connMs":50,"linkMbps":0}' ;;
	rtt60) cfg='{"rttMs":60,"mbps":100,"jitterMs":0,"connMs":120,"linkMbps":0}' ;;
	rtt25link500) cfg='{"rttMs":25,"mbps":200,"jitterMs":0,"connMs":50,"linkMbps":500}' ;;
	*) echo "unknown profile $profile" >&2; exit 1 ;;
	esac
	curl -sf -X POST -H 'content-type: application/json' -d "$cfg" "http://127.0.0.1:$PORT/__config" > /dev/null
	matrix "$profile"
done
if [ "$R2S3" = 1 ]; then matrix r2s3; fi
