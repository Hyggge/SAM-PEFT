## ViT-Adapter 环境安装
- python==3.9.18
- 所需软件包的安装命令如下
```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.21.1
pip install scipy

ln -s ../detection/ops ./
cd ops
sh make.sh # compile deformable attention (srun -p normal -w irip-c3-compute-3 --gres=gpu:3080ti:1   -c 16 --mem 16G sh make.sh 必须是 compute-3 节点)
```

如果出现了 FormatCode() got an unexpected keyword argument 'verify', 只需要 `pip install yapf==0.40.1` 即可

## ViT-Adapter-dcnv3 环境安装
- python==3.10.12
- 所需软件包的安装命令如下
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.21.1
pip install scipy
pip install https://github.com/OpenGVLab/InternImage/releases/download/whl_files/DCNv3-1.0+cu113torch1.11.0-cp310-cp310-linux_x86_64.whl

ln -s ../detection/ops ./
cd ops
sh make.sh # compile deformable attention (srun -p normal -w irip-c3-compute-3 --gres=gpu:3080ti:1   -c 16 --mem 16G sh make.sh 必须是 compute-3 节点)
```
