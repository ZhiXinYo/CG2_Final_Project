import mmengine
from tqdm import tqdm
import os


def convert_nusc_to_coco(ann_file, out_file):
    # 加载 .pkl 文件
    data = mmengine.load(ann_file)
    data_list = data['data_list']

    # 提取类别信息
    categories = [{'id': i, 'name': name} for i, name in enumerate(data['metainfo']['categories'])]

    images = []
    annotations = []
    ann_id = 0

    print(f"开始转换 {ann_file} ...")
    for sample_idx, info in enumerate(tqdm(data_list)):
        # 遍历 6 个摄像头
        for cam_type, cam_info in info['images'].items():

            # 【核心修改】直接使用 img_path 作为 ID
            # 这与 Det3DDataset 基类中 metainfo['img_path'] 里的字符串完全一致
            img_path = cam_info['img_path']
            img_id = img_path

            # 注意：如果你的 data_prefix 在读取时会自动补全绝对路径，
            # 这里的 img_id 建议保持为 .pkl 里的相对路径，这样最稳妥。
            images.append({
                'id': img_id,  # 作为唯一标识符
                'file_name': img_path,  # 图片相对路径
                'height': 900,
                'width': 1600
            })

            # 提取该摄像头的 2D GT (来自 3D 投影得到的 bbox)
            if 'cam_instances' in info and cam_type in info['cam_instances']:
                for inst in info['cam_instances'][cam_type]:
                    bbox = inst['bbox']  # [x1, y1, x2, y2]
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]

                    # 过滤无效框
                    if w <= 0 or h <= 0:
                        continue

                    annotations.append({
                        'id': ann_id,
                        'image_id': img_id,  # 对应上面的 img_id
                        'category_id': inst['bbox_label'],
                        'bbox': [bbox[0], bbox[1], w, h],  # COCO 格式 [x, y, w, h]
                        'area': float(w * h),
                        'iscrowd': 0
                    })
                    ann_id += 1

    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 保存 JSON
    mmengine.dump(coco_format, out_file)
    print(f"成功保存 2D JSON 标注至: {out_file}")


if __name__ == '__main__':
    convert_nusc_to_coco('data/nuscenes/nuscenes_infos_val.pkl', 'data/nuscenes/nuscenes_2d_val.json')