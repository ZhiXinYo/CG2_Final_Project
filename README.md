
# Fast Segment Anything
Original github [FastSAM Official code Repository](https://github.com/CASIA-LMC-Lab/FastSAM)

## Installation

Create the conda env. The code requires `python>=3.7`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.

```shell
conda create -n FastSAM python=3.9
conda activate FastSAM
```

Install the packages:

```shell
cd FastSAM
pip install -r requirements.txt
```

Install CLIP(Required if the text prompt is being tested.):

```shell
pip install git+https://github.com/openai/CLIP.git
```

Download a [model checkpoint](#model-checkpoints).

## Run python code to process image

Run process.py
```shell
python process.py 
```

Necessary Paths and Class Settings:
```shell
MODEL_WEIGHT = "./FastSAM-x.pt"
IMAGE_DIR = "/root/autodl-tmp/data/image/training/image_3"    # 原始图片文件夹
CATEGORIES = ["DontCare", "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]  # 自定义类别列表

```

## Results

Some examples of results are saved in ./ouput folder.