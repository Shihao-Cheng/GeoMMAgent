#!/usr/bin/env bash
# From GeoMMAgent repo root. Override parquet: PARQUET=datasets/my.parquet bash run_benchmark.sh
set -euo pipefail
cd "$(dirname "$0")"

PARQUET="${PARQUET:-datasets/validation.parquet}"
WORKERS="${WORKERS:-20}"

python run/run_benchmark_parallel.py --parquet "$PARQUET" --workers "$WORKERS"
