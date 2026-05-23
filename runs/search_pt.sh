#!/usr/bin/env bash
set -euo pipefail

workers="${1:-15}"
shift || true

for _ in $(seq 1 "${workers}"); do
  python hyperopt/scripts/large_batch_search.py --stage pretrain --pretrain_trials 100 "$@" &
done

wait
