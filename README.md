## 1. 环境配置 (Environment Setup)

请务必按顺序安装，确保依赖完整：

```bash
# 1. 创建基础环境
conda create -n pointpainting python=3.8 -y
conda activate pointpainting

# 2. 安装 PyTorch (根据你的 CUDA 版本调整，推荐 CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 OpenMMLab 基础组件
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

# 4. 安装 MMDetection3D (源码安装，方便后续改模型代码)
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .

# 5. [关键] 安装 MMSegmentation 及相关依赖
# 我们需要调用 mmseg 的接口来跑分割网络
pip install "mmsegmentation>=1.0.0"
pip install ftfy regex  # DeepLab 等模型需要的文本处理依赖

# 6. 安装可视化依赖
pip install open3d matplotlib tqdm
```

## 2. 准备模型与配置 (Prepare Data)

为了解决配置文件继承（`_base_`）的问题，将 mmsegmentation 的配置文件夹独立复制到了项目中。

**请运行项目根目录下的自动脚本，一键完成模型下载和配置提取：**

```bash
# 确保在 mmdetection3d 目录下
bash get_deeplabv3plus.sh
```

运行后，你的目录结构应如下所示：

```text
mmdetection3d/
├── checkpoints/                   
│   └── deeplabv3plus_xxx.pth  <-- [模型权重]
├── configs_seg/                           <-- [分割配置] 从 mmseg 提取的完整配置包
│   ├── _base_/ ...
│   └── deeplabv3plus/ ...
├── data/
│   └── kitti/ ...                         <-- 数据集放这里
├── painting_mmdet3d.py             <-- [核心脚本]
└── ...
```

## 3. 运行 Painting (Run)

**生成增强点云：**
```bash
python painting_mmdet3d.py
```
*   **功能**：读取 Kitti 图片和雷达，生成包含语义特征的新点云。
*   **输出**：`data/kitti/training/velodyne_painted_stereo/`


## 4. TODO

*   **Model Training**: 修改 PointPillars Config，尝试训练
*   **Innovation**: 修改 `painting_mmdet3d.py`，优化算法