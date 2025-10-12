#!/bin/bash
#PBS -P rp06
#PBS -q gpuvolta
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l mem=16GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -l storage=scratch/rp06
#PBS -l jobfs=2GB

module load cuda/12.6.2
module load ffmpeg/4.1.3
source /scratch/rp06/sl5952/VC-SA/.venv/bin/activate
export HF_HOME="/scratch/rp06/sl5952/VC-SA/.cache"
export HF_HUB_OFFLINE=1
# eva_giant_patch14_224.clip_ft_in1k
cd ../..
python3 -m src.run --dataset hmdb51 --data-root ./datasets/hmdb51 --freeze-backbone --vit-name eva_giant_patch14_224.clip_ft_in1k --frames-per-clip 12 --frieren-num-dirs 10 --batch-size 2 --num-workers 4 --precision 32 --no-future-warning --no-user-warning --use-test-as-val --no-tqdm --print-interval 1000 >> M019.log 2>&1
