#!/usr/bin/env bash
set -euo pipefail

workers="${1:-16}"
shift || true

for _ in $(seq 1 "${workers}"); do
  python hyperopt/scripts/finetune_with_pretrain_options.py "$@" &
done

wait
