#!/usr/bin/env bash
# Hybride fixe : base axonale initialisee ET gelee a zero, offsets conserves.
# Reference : synaptique a delais fixes uniformes entiers, comme les offsets
# hybrides : 0..hybrid_offset_scale inclus (borne de perf_HAR.py).
# Les poids restent appris. Meme distribution, pas necessairement meme tirage.
# Par defaut : hybride seul, seed d'entrainement 0, offsets seeds 123/124/125.
# Personnaliser : TRAIN_SEED=0 OFFSET_SEEDS="123 124 125" bash experiments/scripts/test_HAR_hybrid_zero.sh
# Pour relancer aussi la reference : INCLUDE_SYNAPTIC=1 bash experiments/scripts/test_HAR_hybrid_zero.sh
# bash experiments/scripts/test_HAR_hybrid_zero.sh
# CUDA_VISIBLE_DEVICES=2 bash experiments/scripts/test_HAR_hybrid_zero.sh
# Apercu : DRY_RUN=1 bash experiments/scripts/test_HAR_hybrid_zero.sh
# Test court : OFFSET_SEEDS=123 SMOKE=1 bash experiments/scripts/test_HAR_hybrid_zero.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=${PYTHON:-python}
seed=${TRAIN_SEED:-0}
read -r -a offset_seeds <<< "${OFFSET_SEEDS:-123 124 125}"
ROOT=${OUT_ROOT:-exp/HAR/hybrid_zero/$(date +%Y-%m-%d_%H-%M-%S)_$$}
models=(
    SNN_fixed_hybrid_recurrent_delays
)
if [[ ${INCLUDE_SYNAPTIC:-0} == 1 ]]; then
    models+=(SNN_fixed_synaptic_recurrent_delays)
fi

for offset_seed in "${offset_seeds[@]}"; do
    for model in "${models[@]}"; do
        # La reference ne depend pas de la seed des offsets : un seul run suffit.
        if [[ $model == SNN_fixed_synaptic_recurrent_delays ]]; then
            [[ $offset_seed == "${offset_seeds[0]}" ]] || continue
            out="$ROOT/$model/seed$seed"
        else
            out="$ROOT/$model/offset_seed$offset_seed/seed$seed"
        fi
        cmd=("$PYTHON" -u experiments/train.py --dataset har --model "$model"
            --seeds "$seed" --save-delay-history --delay-history-interval 10
            --out "$out")
        if [[ $model == SNN_fixed_hybrid_recurrent_delays ]]; then
            cmd+=(--init-rec-delay half_normal --delay-std-init 0
                --hybrid-delay-seed "$offset_seed")
        else
            cmd+=(--init-rec-delay uniform_integer)
        fi
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
