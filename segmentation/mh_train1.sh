CONFIG=$1

srun -p normal -w irip-c3-compute-3 --gres=gpu:3080ti:1  -c 8 --mem 32G \
python -u  train.py ${CONFIG}  --work-dir  work/temp1  --seed 316738450 --resume-from work/temp1/iter_80000.pth
