import torch
import mmcv
import os
import numpy as np
from mmengine import Config
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.registry import VISUALIZERS
from mmdet3d.utils import register_all_modules

register_all_modules()

def get_lidar2img_matrix(calib_path):
    """从KITTI标定文件中计算点云到图像的投影矩阵，并返回 4x4 格式"""
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
    R0 = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    
    v2c_line = [l for l in lines if 'Tr_velo' in l][0]
    V2C = np.array([float(x) for x in v2c_line.split()[1:]]).reshape(3, 4)
    
    R0_4x4 = np.eye(4)
    R0_4x4[:3, :3] = R0
    V2C_4x4 = np.eye(4)
    V2C_4x4[:3, :4] = V2C
    
    # 计算初步投影矩阵 (3x4)
    lidar2img_3x4 = P2 @ R0_4x4 @ V2C_4x4
    
    # 扩展为 4x4 矩阵
    lidar2img_4x4 = np.eye(4)
    lidar2img_4x4[:3, :4] = lidar2img_3x4
    
    return lidar2img_4x4

def run_save_visualize(sample_idx, config_path, checkpoint_path, out_dir, method_name='ours'):
    model = init_model(config_path, checkpoint_path, device='cuda:2')
    data_root = 'data/kitti_painted/' 
    if method_name == 'ours':
        pcd_path = os.path.join(data_root, 'training/velodyne_painted_1225', f'{sample_idx}.bin')
    else:
        pcd_path = os.path.join('data/kitti/', 'training/velodyne', f'{sample_idx}.bin')
        
    img_path = os.path.join(data_root, 'training/image_2', f'{sample_idx}.png')
    calib_path = os.path.join(data_root, 'training/calib', f'{sample_idx}.txt')
    
    result = inference_detector(model, pcd_path)
    if isinstance(result, (list, tuple)):
        result = result[0]

    lidar2img = get_lidar2img_matrix(calib_path)
    result.set_metainfo({'lidar2img': lidar2img})
    
    if 'save_dir' not in model.cfg.visualizer:
        model.cfg.visualizer['save_dir'] = out_dir
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    
    img = mmcv.imread(img_path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    data_input = dict(img=img)
    visualizer.add_datasample(
        'result',
        data_input, # 传入字典
        data_sample=result,
        draw_gt=False,
        pred_score_thr=0.35,
        show=False
    )

    vis_img = visualizer.get_image()
    tag = 'ours' if ('drop' in config_path or 'painted' in config_path) else 'base'
    save_filename = f"vis_{sample_idx}_{tag}.png"
    out_path = os.path.join(out_dir, save_filename)
    
    mmcv.imwrite(mmcv.imconvert(vis_img, 'rgb', 'bgr'), out_path)
    print(f"saved to: {out_path}")
        
if __name__ == '__main__':
    # 图片ID
    samples = ['000008', '000120', '000134']     
    ours_cfg = 'configs/pointpillars/pointpainting_vis.py'
    ours_ckpt = 'work_dirs/pointpillars_kitti_painted_1225/epoch_90.pth'
    
    output_folder = 'case_study_results'
    os.makedirs(output_folder, exist_ok=True)
    
    for s in samples:
        run_save_visualize(s, ours_cfg, ours_ckpt, output_folder)