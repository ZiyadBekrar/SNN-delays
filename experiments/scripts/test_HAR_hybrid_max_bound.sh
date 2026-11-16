#!/usr/bin/env bash
# Hybride recurrent appris : balayage de la borne inclusive des offsets synaptiques.
# Par defaut : bornes 0/1/2/4/8, trois seeds d'entrainement 0/1/2 par borne.
# bash experiments/scripts/test_HAR_hybrid_max_bound.sh
# MAX_BOUNDS="0 2 4 8" SEEDS="0 1" DELAY_STD_INIT=0 bash experiments/scripts/test_HAR_hybrid_max_bound.sh
# Apercu : DRY_RUN=1 bash experiments/scripts/test_HAR_hybrid_max_bound.sh
# Test court : MAX_BOUNDS=4 SEEDS=0 SMOKE=1 bash experiments/scripts/test_HAR_hybrid_max_bound.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=${PYTHON:-python}
read -r -a max_bounds <<< "${MAX_BOUNDS:-0 1 2 4 8}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
std=${DELAY_STD_INIT:-2}
model=SNN_recurrent_hybrid_delays
ROOT=${OUT_ROOT:-exp/HAR/hybrid_max_bound/$(date +%Y-%m-%d_%H-%M-%S)_$$}

for max_bound in "${max_bounds[@]}"; do
    for seed in "${seeds[@]}"; do
        out="$ROOT/$model/max_bound$max_bound/std$std/seed$seed"
        cmd=("$PYTHON" -u experiments/train.py --dataset har --model "$model"
            --seeds "$seed" --init-rec-delay half_normal --delay-std-init "$std"
            --hybrid-offset-scale "$max_bound"
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
