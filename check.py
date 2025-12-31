import numpy as np
import pickle
import os
from tqdm import tqdm

# 检查painted点云数据的完整性
DATA_ROOT = 'data/kitti_painted/'

def check_bin_file_dimension(file_path, expected_dim=8):
    if not os.path.exists(file_path):
        print(f"  [ERROR] File not found: {file_path}")
        return False, 0
        
    try:
        points = np.fromfile(file_path, dtype=np.float32)
        total_elements = points.shape[0]
        
        if total_elements == 0:
            return True, 0
            
        if total_elements % expected_dim == 0:
            return True, total_elements
        else:
            return False, total_elements
            
    except Exception as e:
        print(f"  [ERROR] Could not read file {file_path}: {e}")
        return False, 0


def main():
    print("===== Starting Data Integrity Check for PointPainting =====")
    
    # --- 1. 检查主场景点云 ---
    scene_velodyne_path = os.path.join(DATA_ROOT, 'training', 'velodyne_painted')
    print(f"\n[Phase 1] Checking main scene point clouds in: {scene_velodyne_path}")
    
    if not os.path.isdir(scene_velodyne_path):
        print(f"  [FATAL] Directory not found: {scene_velodyne_path}")
        return

    scene_files = sorted([f for f in os.listdir(scene_velodyne_path) if f.endswith('.bin')])
    bad_scene_files = []
    
    for filename in tqdm(scene_files, desc="Scanning scenes"):
        file_path = os.path.join(scene_velodyne_path, filename)
        is_ok, num_elements = check_bin_file_dimension(file_path)
        if not is_ok:
            bad_scene_files.append((file_path, num_elements))
            
    if not bad_scene_files:
        print("  [SUCCESS] All main scene point cloud files are valid 8D files.")
    else:
        print(f"  [WARNING] Found {len(bad_scene_files)} invalid main scene files:")
        for path, elements in bad_scene_files:
            print(f"    - Path: {path}, Total Elements: {elements} (not divisible by 8)")
            
    dbinfo_path = os.path.join(DATA_ROOT, 'kitti_dbinfos_train_painted.pkl')
    print(f"\n[Phase 2] Checking GT database point clouds listed in: {dbinfo_path}")
    
    if not os.path.exists(dbinfo_path):
        print(f"  [FATAL] DB info file not found: {dbinfo_path}. Cannot check database.")
        return
        
    with open(dbinfo_path, 'rb') as f:
        db_infos = pickle.load(f)
        
    bad_db_files = []
    total_db_files = 0
    
    for class_name, infos_list in db_infos.items():
        total_db_files += len(infos_list)
        for info in tqdm(infos_list, desc=f"Scanning DB '{class_name}'"):
            if 'path' not in info:
                print(f"  [WARNING] No 'path' key in db_info for an object of class {class_name}. Skipping.")
                continue
            
            relative_path = info['path']
            file_path = os.path.join(DATA_ROOT, relative_path)
            
            is_ok, num_elements = check_bin_file_dimension(file_path)
            if not is_ok:
                bad_db_files.append((file_path, num_elements))

    if not bad_db_files:
        print(f"  [SUCCESS] All {total_db_files} GT database point cloud files are valid 8D files.")
    else:
        print(f"  [WARNING] Found {len(bad_db_files)} invalid GT database files:")
        for path, elements in bad_db_files:
            print(f"    - Path: {path}, Total Elements: {elements} (not divisible by 8)")
            
    print("\n===== Check Finished =====")
    if not bad_scene_files and not bad_db_files:
        print("Congratulations! All checked data files appear to be correctly formatted as 8D.")
    else:
        print("Please review the warnings above. These invalid files are likely the cause of the reshape error.")


if __name__ == '__main__':
    main()