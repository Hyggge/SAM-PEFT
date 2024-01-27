CONFIG=$1

srun -p normal -w irip-c3-compute-1 --gres=gpu:3090:1 --ntasks=1  -c 16 --mem 32G \
python -u  train.py ${CONFIG}  --work-dir  work/temp --launcher="slurm" --seed 316738450
