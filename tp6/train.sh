#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:20:00
#SBATCH -J tp6-train
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
# Meilleure configuration de l'exercice 3 : bf16 + channels_last + compile, images redimensionnées, 3 workers
python -u train.py --tag best --precision bf16 --channels-last --compile --data small --workers 3 --epochs 2