CONFIG=$1

srun -p normal -w irip-c3-compute-1 --gres=gpu:3080ti:1  -c 8 --mem 32G \
python -u  train.py ${CONFIG}  --work-dir  work/temp7_whole  --seed 316738450
