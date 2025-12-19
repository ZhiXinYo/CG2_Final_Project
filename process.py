# from fastsam import FastSAM, FastSAMPrompt

# model = FastSAM('./FastSAM-x.pt')
# IMAGE_PATH = '/root/autodl-tmp/data/image/check/000001.png'
# DEVICE = 'cuda'
# everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9,)
# prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)

# # # everything prompt
# # ann = prompt_process.everything_prompt()

# # prompt_process.plot(annotations=ann,output_path='./output/0.jpg',)

# # text prompt
# ann = prompt_process.text_prompt(text='car')
# # class_masks, pixel_prob, class_clip_scores = prompt_process.text_prompt_multi_classes(['car'])

# # # point prompt
# # # points default [[0,0]] [[x1,y1],[x2,y2]]
# # # point_label default [0] [1,0] 0:background, 1:foreground
# # ann = prompt_process.point_prompt(points=[[620, 360]], pointlabel=[1])

# prompt_process.plot(annotations=ann,output_path='./output/0.jpg',)


import os
import cv2
import numpy as np
import torch
# from fastsam import FastSAM  # 假设你有FastSAM主模型类
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
# from your_module import FastSAMPrompt  # 替换为你的实际模块名
from fastsam import FastSAM, FastSAMPrompt

# -------------------------- 配置参数 --------------------------
MODEL_WEIGHT = "./FastSAM-x.pt"
IMAGE_DIR = "/root/autodl-tmp/data/image/check"    # 原始图片文件夹
MASK_OUTPUT_DIR = "./output_masks"  # 逐类别mask输出
SCORE_OUTPUT_DIR = "./output_scores"  # H×W×C概率输出
CATEGORIES = ["DontCare", "Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]  # 自定义类别列表
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
NUM_WORKERS = 1  # 线程数
CONF_THRESH = 0.25
IOU_THRESH = 0.7
# CONF_THRESH = 0.4
# IOU_THRESH = 0.9

# -------------------------- 初始化 --------------------------
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(SCORE_OUTPUT_DIR, exist_ok=True)
fastsam_model = FastSAM(MODEL_WEIGHT)

# -------------------------- 单张图片处理函数 --------------------------
def process_single_image(img_path):
    img_name = os.path.basename(img_path) 
    img_basename = os.path.splitext(img_name)[0] 

    results = fastsam_model(
        img_path, 
        device=DEVICE,
        retina_masks=True,
        imgsz=1024, 
        conf=CONF_THRESH, 
        iou=IOU_THRESH,
    )

    # 初始化FastSAMPrompt
    prompt_processor = FastSAMPrompt(img_path, results, device=DEVICE)

    # 获取H×W×C概率分布
    class_masks, pixel_prob, class_clip_scores = prompt_processor.text_prompt_multi_classes(CATEGORIES)

    # 保存逐类别mask
    for cat, mask in class_masks.items():
        mask_path = os.path.join(MASK_OUTPUT_DIR, f"{img_basename}_{cat}.png")
        # 转为二值图保存
        mask_8bit = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_8bit)

    # 保存H×W×C概率矩阵（npy格式，可直接加载）
    prob_npy_path = os.path.join(SCORE_OUTPUT_DIR, f"{img_basename}_prob.npy")
    np.save(prob_npy_path, pixel_prob)

    # 保存类别CLIP分数（JSON）
    import json
    score_json_path = os.path.join(SCORE_OUTPUT_DIR, f"{img_basename}_clip_scores.json")
    with open(score_json_path, "w") as f:
        json.dump(class_clip_scores, f, indent=4)

    return img_path, "success"

# -------------------------- 批量执行 --------------------------
if __name__ == "__main__":
    # 获取所有图片路径
    img_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    # 多线程批量处理
    print(f"处理 {len(img_paths)}张图片...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_single_image, img_paths),
            total=len(img_paths),
            desc="处理进度"
        ))

    # 输出结果统计
    success = sum([1 for _, status in results if status == "success"])
    print(f"\n 处理完成：成功 {success}/{len(img_paths)}")
    print(f" Mask保存至：{MASK_OUTPUT_DIR}")
    print(f" H×W×C概率矩阵保存至：{SCORE_OUTPUT_DIR}（.npy）")

    # 验证输出维度
    sample_npy = os.path.join(SCORE_OUTPUT_DIR, f"{os.path.splitext(os.listdir(IMAGE_DIR)[0])[0]}_prob.npy")
    if os.path.exists(sample_npy):
        prob = np.load(sample_npy)

