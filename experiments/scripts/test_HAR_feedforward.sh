#!/usr/bin/env bash
# HAR : delais feedforward appris axonaux, synaptiques et hybrides.
# 3 modeles x 3 seeds = 9 runs, avec les hyperparametres de configs/perf_HAR.py.
# bash experiments/scripts/test_HAR_feedforward.sh
# Apercu : DRY_RUN=1 bash experiments/scripts/test_HAR_feedforward.sh
# Test court : SEEDS=0 SMOKE=1 bash experiments/scripts/test_HAR_feedforward.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=${PYTHON:-python}
read -r -a seeds <<< "${SEEDS:-0 1 2}"
ROOT=${OUT_ROOT:-exp/HAR/feedforward/$(date +%Y-%m-%d_%H-%M-%S)_$$}
models=(
    SNN_axonal_feedforward_delays
    SNN_synaptic_feedforward_delays
    SNN_hybrid_feedforward_delays
)

for model in "${models[@]}"; do
    for seed in "${seeds[@]}"; do
        out="$ROOT/$model/seed$seed"
        cmd=("$PYTHON" -u experiments/train.py --dataset har --model "$model"
            --seeds "$seed" --out "$out")
        [[ -z ${EPOCHS:-} ]] || cmd+=(--epochs "$EPOCHS")
        [[ ${SMOKE:-0} != 1 ]] || cmd+=(--smoke)
        printf '%q ' "${cmd[@]}"; printf '\n'
        if [[ ${DRY_RUN:-0} != 1 ]]; then
            if [[ -e "$out" ]]; then
                echo "Dossier deja present : $out ; choisir un autre OUT_ROOT." >&2
                exit 1
            fi
            mkdir -p "$out"
            "${cmd[@]}" 2>&1 | tee "$out/console.log"
        fi
    done
done
