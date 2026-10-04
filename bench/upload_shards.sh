#!/usr/bin/env bash
# Uploads data/prep/ingest/*.f32 to R2 as ingest/XXXXX.f32, skipping ones already present.
# Usage: bench/upload_shards.sh [parallelism]
set -euo pipefail
BUCKET=vib-bench
PAR=${1:-6}
cd "$(dirname "$0")/../data/prep/ingest"
ls *.f32 | xargs -P "$PAR" -I{} bash -c '
  f="{}"; key="ingest/$f"
  size=$(stat -c %s "$f")
  if cf r2 objects put "$key" --bucket-name '"$BUCKET"' --file "$f" --content-type application/octet-stream -q >/dev/null 2>&1; then
    echo "ok $key $size"
  else
    echo "FAIL $key" >&2
  fi
'
