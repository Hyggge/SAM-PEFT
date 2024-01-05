CONFIG=$1
CKPT=$2

srun -p normal -w irip-c3-compute-1 --gres=gpu:3090:1   -c 16 --mem 16G \
python test.py ${CONFIG} ${CKPT}  --eval mIoU --work-dir "work/test/test_log" --out "work/test/test_out.pkl" --show-dir "work/test/test_show"
