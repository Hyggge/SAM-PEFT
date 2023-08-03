CONFIG=$1
CKPT=$2

srun -p normal -w irip-c3-compute-2 --gres=gpu:3080ti:1   -c 16 --mem 16G \
python test.py ${CONFIG} ${CKPT}  --eval mIoU --work-dir "work/test" 
