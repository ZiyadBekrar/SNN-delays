#!/usr/bin/env bash
# Deuxieme experience : axonal, hybride recurrent, synaptique ; une seed chacun.
# Delais sauvegardes apres 10, 20, 30... epoques (indices fichiers 0009, 0019...).
# Les hyperparametres, dont delay_std_init, viennent de configs/perf_HAR.py.
# bash experiments/scripts/test_HAR_delay_history.sh
# Apercu : DRY_RUN=1 bash experiments/scripts/test_HAR_delay_history.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PYTHON=${PYTHON:-python}
seed=${SEED:-0}
ROOT=${OUT_ROOT:-exp/HAR/delay_history/$(date +%Y-%m-%d_%H-%M-%S)_$$}
models=(SNN_axonal_recurrent_delays SNN_recurrent_hybrid_delays
        SNN_synaptic_recurrent_delays)

for model in "${models[@]}"; do
    out="$ROOT/$model/seed$seed"
    cmd=("$PYTHON" -u experiments/train.py --dataset har --model "$model"
        --seeds "$seed" --save-delay-history --delay-history-interval 10 --out "$out")
    [[ -z ${EPOCHS:-} ]] || cmd+=(--epochs "$EPOCHS")
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
