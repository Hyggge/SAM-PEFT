CONFIG=$1
CKPT=$2
IMG=$3

srun -p normal -w irip-c3-compute-2 --gres=gpu:3080ti:1   -c 16 --mem 16G  python image_demo.py \
	${CONFIG} ${CKPT} ${IMG} 
