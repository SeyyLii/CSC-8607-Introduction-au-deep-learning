#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:12:00
#SBATCH -J tp6-ex3b
#SBATCH -o logs/%x-%j.out

# Vérification : débit RÉEL (temps d'une époque) quand le chargement limite.
# La médiane par pas de bench.py est biaisée à 3 workers (batchs livrés par rafales).
set -euo pipefail
source tp6_env.sh
for D in raw small; do
    python -u bench.py --tag bf16-cl-$D-w3-epoch --precision bf16 --channels-last \
           --data $D --workers 3 --repeat 1 --epoch
    python -u bench.py --tag bf16-cl-compile-$D-w3-epoch --precision bf16 --channels-last --compile \
           --data $D --workers 3 --repeat 1 --epoch
done