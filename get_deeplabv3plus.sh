#!/bin/bash

echo "=========================================="
echo "   PointPainting Environment Preparation"
echo "=========================================="

# 1. 准备权重文件夹
if [ ! -d "checkpoints" ]; then
    mkdir checkpoints
    echo "[+] Created checkpoints directory."
fi

# 2. 下载 DeepLabV3+ 权重 (Cityscapes 80k)
MODEL_URL="https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth"
SAVE_PATH="checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth"

if [ ! -f "$SAVE_PATH" ]; then
    echo "[*] Downloading Segmentation Model Weights..."
    wget $MODEL_URL -O $SAVE_PATH
    echo "[+] Model saved to $SAVE_PATH"
else
    echo "[.] Model weights already exist."
fi

# 3. 准备 Configs 文件 (configs_seg)
# if [ ! -d "configs_seg" ]; then
#     echo "[*] Downloading MMSegmentation Configs..."
    
#     # 下载 mmsegmentation main 分支的 zip
#     wget https://github.com/open-mmlab/mmsegmentation/archive/refs/heads/main.zip -O mmseg_temp.zip
    
#     echo "[*] Extracting configs..."
#     # 解压 zip 中的 configs 文件夹
#     unzip -q mmseg_temp.zip "mmsegmentation-main/configs/*"
    
#     # 移动并重命名
#     mv mmsegmentation-main/configs configs_seg
    
#     # 清理垃圾文件
#     rm mmseg_temp.zip
#     rm -rf mmsegmentation-main
    
#     echo "[+] Configs extracted to 'configs_seg/' directory."
# else
#     echo "[.] 'configs_seg' directory already exists."
# fi

echo "=========================================="
echo "   All Set! Ready for Painting."
echo "=========================================="