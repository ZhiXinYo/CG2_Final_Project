## 1. 环境配置 (Environment Setup)

请务必按顺序安装，确保依赖完整（参考[MMDetection3D 官方文档](https://mmdetection3d.readthedocs.io/zh-cn/latest/get_started.html)）：

```bash
# 1. 创建基础环境
conda create -n pointpainting python=3.8 -y
conda activate pointpainting

# 2. 安装 PyTorch (根据你的 CUDA 版本调整，推荐 CUDA 12.1)
conda install pytorch torchvision -c pytorch

# 3. 安装 OpenMMLab 基础组件
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4"
mim install "mmdet>=3.0.0"

# 4. 安装 MMDetection3D (源码安装，方便后续改模型代码)
git clone https://github.com/open-mmlab/mmdetection3d.git -b dev-1.x
cd mmdetection3d
pip install -v -e .

# 5. [关键] 安装 MMSegmentation 及相关依赖
# 我们需要调用 mmseg 的接口来跑分割网络
pip install "mmsegmentation>=1.0.0"
pip install ftfy regex  # DeepLab 等模型需要的文本处理依赖

# 6. 安装可视化依赖（可选）
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
python paint_kitti.py
```
*   **功能**：读取 Kitti 图片和雷达，生成包含语义特征的新点云（8D 数据）。
*   **输出**：`data/kitti_painted/training/your_filename/`


## 4. 构建GT_DataBase（适配 8D 数据）
确保已经按照[mmdet3d官方文档](https://mmdet3dtai.readthedocs.io/en/latest/datasets/kitti_det.html)处理好Kitti数据集结构

执行以下命令构建GT_DataBase（确保路径都正确）
```bash
python tools/create_painted_database.py
```

## 5. 执行训练
修改配置文件 `configs/pointpillars/pointpainting.py`，确保读取pkl的路径正确（修改为上一步新生成的pkl路径）

目前设置的是只训练，不评估。也可以添加评估设置，一步到位，参考原始配置文件 `configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py` 添加val和test的dataloader、evaluator、cfg

```bash
python tools/train.py configs/pointpillars/pointpainting.py --work-dir work_dirs/xxx
```

------

## 6. （改进）Yolo + PaintingPointPillars
1. 安装mmyolo。
```bash
mim install "mmyolo"
```
2. 按照mmdet3d的格式预处理nus数据集。
3. 对val标签进行预处理。
```bash
python tools/nus2coco.py
```
4. 按需调整configs/pointpainting/yolo_painter_nus.py配置文件。（调整version为v1.0-trainval, img_scale优先选择大一点的，实在不行选择小的, batch_size看情况调整）
5. 执行训练命令。
```bash
python tools/train.py configs/pointpainting/yolo_painter_nus.py
```

## 7. （改进）稀疏采样
实现在 `my_transforms.py` 中
训练配置：`configs/pointpillars/pointpainting_plus.py`