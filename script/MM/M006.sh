#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=10:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -l jobfs=2GB

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/VC-SA/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/VC-SA/.cache"
export HF_HUB_OFFLINE=1
# aimv2_large_patch14_224.apple_pt_dist
cd ../..
python3 -m src.run --dataset hmdb51 --data-root ./datasets/hmdb51 --freeze-backbone --vit-name aimv2_large_patch14_224.apple_pt_dist --frames-per-clip 12 --frida-num-dirs 10 --batch-size 2 --num-workers 4 --precision 32 --no-future-warning --no-user-warning --use-test-as-val --no-tqdm --print-interval 1000 >> M006.log 2>&1
