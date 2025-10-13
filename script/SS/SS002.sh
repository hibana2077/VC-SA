#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=20GB
#PBS -l walltime=10:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -l jobfs=8GB

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/VC-SA/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/VC-SA/.cache"
export HF_HUB_OFFLINE=1
# tnt_s_patch16_224.in1k
cd ../..
python3 -m src.run --dataset ssv2 --data-root ./datasets/ssv2 --freeze-backbone --vit-name tnt_s_patch16_224.in1k --frames-per-clip 12 --frieren-num-dirs 6 --batch-size 2 --num-workers 8 --precision 32 --no-future-warning --no-user-warning --use-test-as-val --no-tqdm --print-interval 10000 >> SS002.log 2>&1
