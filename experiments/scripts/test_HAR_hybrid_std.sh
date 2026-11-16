#!/usr/bin/env bash
# Premiere experience : hybride recurrent appris, stds 0/1/2, seeds 0/1/2.
# bash experiments/scripts/test_HAR_hybrid_std.sh
# Apercu : DRY_RUN=1 bash experiments/scripts/test_HAR_hybrid_std.sh
# Test court : SMOKE=1 bash experiments/scripts/test_HAR_hybrid_std.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=${PYTHON:-python}
read -r -a stds <<< "${STDS:-0 1 2}"
seeds=(0 1 2)
model=SNN_recurrent_hybrid_delays
ROOT=${OUT_ROOT:-exp/HAR/hybrid_std/$(date +%Y-%m-%d_%H-%M-%S)_$$}

# 3 stds x 3 seeds = 9 runs ; historique des delais a chaque epoque.
for std in "${stds[@]}"; do
    for seed in "${seeds[@]}"; do
        out="$ROOT/$model/std$std/seed$seed"
        cmd=("$PYTHON" experiments/train.py --dataset har --model "$model"
            --seeds "$seed" --delay-std-init "$std"
            --save-delay-history --out "$out")
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
