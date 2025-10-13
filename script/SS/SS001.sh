#!/bin/bash
#PBS -P rp06
#PBS -q dgxa100
#PBS -l ngpus=1
#PBS -l ncpus=16
#PBS -l mem=24GB
#PBS -l walltime=10:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -l jobfs=8GB

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/VC-SA/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/VC-SA/.cache"
export HF_HUB_OFFLINE=1
# beitv2_large_patch16_224.in1k_ft_in22k
cd ../..
python3 -m src.run --dataset ssv2 --data-root ./datasets/ssv2 --freeze-backbone --vit-name beitv2_large_patch16_224.in1k_ft_in22k --frames-per-clip 12 --frieren-num-dirs 10 --batch-size 2 --num-workers 8 --precision 32 --no-future-warning --no-user-warning --use-test-as-val --no-tqdm --print-interval 10000 >> SS001.log 2>&1
