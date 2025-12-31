import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
from PIL import Image
from tqdm import tqdm

# from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

# import pcdet_utils.calibration_kitti as calibration_kitti

from sklearn.neighbors import NearestNeighbors

TRAINING_PATH = "data/kitti/training/"
TWO_CAMERAS = True
SEG_NET_OPTIONS = ["deeplabv3", "deeplabv3plus", "hma", "segformer"]
# TODO choose the segmentation network you want to use, deeplabv3 = 0 deeplabv3plus = 1 hma = 2
SEG_NET = 3 #TODO choose your preferred network


class Painter:
    def __init__(self, seg_net_index):
        self.root_split_path = TRAINING_PATH
        self.save_path = TRAINING_PATH + "painted_lidar_geometirc5/"
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.seg_net_index = seg_net_index
        self.model = None
        if seg_net_index == 0:
            print(f'Using Segmentation Network -- {SEG_NET_OPTIONS[seg_net_index]}')
            self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
            self.model.eval()
            if torch.cuda.is_available():
                self.model.to('cuda')
        elif seg_net_index == 1:
            print(f'Using Segmentation Network -- {SEG_NET_OPTIONS[seg_net_index]}')
            config_file = './mmseg/configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
            checkpoint_file = './mmseg/checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth'
            self.model = init_segmentor(config_file, checkpoint_file, device='cuda:0') # TODO edit here if you want to use different device

        
    def get_lidar(self, idx):
        lidar_file = self.root_split_path + 'velodyne/' + ('%s.bin' % idx)
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_score(self, idx, left):
        ''' idx : index string
            left : string indicates left/right camera 
        return:
            a tensor H  * W * 4(deeplab)/5(deeplabv3plus), for each pixel we have 4/5 scorer that sums to 1
        '''
        output_reassign_softmax = None
        if self.seg_net_index == 0:
            filename = self.root_split_path + left + ('%s.png' % idx)
            input_image = Image.open(filename)
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            # move the input and model to GPU for speed if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')

            with torch.no_grad():
                output = self.model(input_batch)['out'][0]

            output_permute = output.permute(1,2,0)
            output_probability,output_predictions =  output_permute.max(2)

            other_object_mask = ~((output_predictions == 0) | (output_predictions == 2) | (output_predictions == 7) | (output_predictions == 15))
            detect_object_mask = ~other_object_mask
            sf = torch.nn.Softmax(dim=2)

            # bicycle = 2 car = 7 person = 15 background = 0
            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1),4)
            output_reassign[:,:,0] = detect_object_mask * output_permute[:,:,0] + other_object_mask * output_probability # background
            output_reassign[:,:,1] = output_permute[:,:,2] # bicycle
            output_reassign[:,:,2] = output_permute[:,:,7] # car
            output_reassign[:,:,3] = output_permute[:,:,15] #person
            output_reassign_softmax = sf(output_reassign).cpu().numpy()

        elif self.seg_net_index == 1:
            filename = self.root_split_path + left + ('%s.png' % idx)
            result = inference_segmentor(self.model, filename)
            # person 11, rider 12, vehicle 13/14/15/16, bike 17/18
            output_permute = torch.tensor(result[0]).permute(1,2,0) # H, W, 19
            sf = torch.nn.Softmax(dim=2)

            output_reassign = torch.zeros(output_permute.size(0),output_permute.size(1), 5)
            output_reassign[:,:,0], _ = torch.max(output_permute[:,:,:11], dim=2) # background
            output_reassign[:,:,1], _ = torch.max(output_permute[:,:,[17, 18]], dim=2) # bicycle
            output_reassign[:,:,2], _ = torch.max(output_permute[:,:,[13, 14, 15, 16]], dim=2) # car
            output_reassign[:,:,3] = output_permute[:,:,11] #person
            output_reassign[:,:,4] = output_permute[:,:,12] #rider
            output_reassign_softmax = sf(output_reassign).cpu().numpy()
        
        elif self.seg_net_index == 2:
            filename = self.root_split_path + "score_hma/" + left + ('%s.npy' % idx)
            output_reassign_softmax = np.load(filename)

        elif self.seg_net_index == 3:
            filename = self.root_split_path + "score_segformer/" + left + ('%s.npy' % idx)
            output_reassign_softmax = np.load(filename)

        return output_reassign_softmax

    def get_calib(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        return calibration_kitti.Calibration(calib_file)
    
    def get_calib_fromfile(self, idx):
        calib_file = self.root_split_path + 'calib/' + ('%s.txt' % idx)
        calib = calibration_kitti.get_calib_from_file(calib_file)
        calib['P2'] = np.concatenate([calib['P2'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['P3'] = np.concatenate([calib['P3'], np.array([[0., 0., 0., 1.]])], axis=0)
        calib['R0_rect'] = np.zeros([4, 4], dtype=calib['R0'].dtype)
        calib['R0_rect'][3, 3] = 1.
        calib['R0_rect'][:3, :3] = calib['R0']
        calib['Tr_velo_to_cam'] = np.concatenate([calib['Tr_velo_to_cam'], np.array([[0., 0., 0., 1.]])], axis=0)
        return calib
    
    def create_cyclist(self, augmented_lidar):
        if self.seg_net_index == 0:
            bike_idx = np.where(augmented_lidar[:,5]>=0.2)[0] # 0, 1(bike), 2, 3(person)
            bike_points = augmented_lidar[bike_idx]
            cyclist_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            for i in range(bike_idx.shape[0]):
                cyclist_mask = (np.linalg.norm(augmented_lidar[:,:3]-bike_points[i,:3], axis=1) < 1) & (np.argmax(augmented_lidar[:,-4:],axis=1) == 3)
                if np.sum(cyclist_mask) > 0:
                    cyclist_mask_total |= cyclist_mask
                else:
                    augmented_lidar[bike_idx[i], 4], augmented_lidar[bike_idx[i], 5] = augmented_lidar[bike_idx[i], 5], 0
            augmented_lidar[cyclist_mask_total, 7], augmented_lidar[cyclist_mask_total, 5] = 0, augmented_lidar[cyclist_mask_total, 7]
            return augmented_lidar
        elif self.seg_net_index == 1 or 2 or 3:
            rider_idx = np.where(augmented_lidar[:,8]>=0.3)[0] # 0, 1(bike), 2, 3(person), 4(rider)
            rider_points = augmented_lidar[rider_idx]
            bike_mask_total = np.zeros(augmented_lidar.shape[0], dtype=bool)
            bike_total = (np.argmax(augmented_lidar[:,-5:],axis=1) == 1)
            for i in range(rider_idx.shape[0]):
                bike_mask = (np.linalg.norm(augmented_lidar[:,:3]-rider_points[i,:3], axis=1) < 1) & bike_total
                bike_mask_total |= bike_mask
            augmented_lidar[bike_mask_total, 8] = augmented_lidar[bike_mask_total, 5]
            augmented_lidar[bike_total^bike_mask_total, 4] = augmented_lidar[bike_total^bike_mask_total, 5]
            return augmented_lidar[:,[0,1,2,3,4,8,6,7]]

    def cam_to_lidar(self, pointcloud, projection_mats):
        """
        Takes in lidar in velo coords, returns lidar points in camera coords

        :param pointcloud: (n_points, 4) np.array (x,y,z,r) in velodyne coordinates
        :return lidar_cam_coords: (n_points, 4) np.array (x,y,z,r) in camera coordinates
        """

        lidar_velo_coords = copy.deepcopy(pointcloud)
        reflectances = copy.deepcopy(lidar_velo_coords[:, -1]) #copy reflectances column
        lidar_velo_coords[:, -1] = 1 # for multiplying with homogeneous matrix
        lidar_cam_coords = projection_mats['Tr_velo2cam'].dot(lidar_velo_coords.transpose())
        lidar_cam_coords = lidar_cam_coords.transpose()
        lidar_cam_coords[:, -1] = reflectances
        
        return lidar_cam_coords
    
    def l1_normalize(self, scores, axis=-1, epsilon=1e-8):
        """
        Normalize scores to sum to 1 along the specified axis (L1 normalization).

        Args:
            scores (np.ndarray): Raw scores of shape (..., C)
            axis (int): Axis along which to normalize (default: last axis)
            epsilon (float): Small value to avoid division by zero

        Returns:
            np.ndarray: Normalized scores with sum = 1 along `axis`
        """
        # Ensure non-negative (optional, but recommended)
        scores = np.clip(scores, a_min=0.0, a_max=None)

        # Compute sum along axis
        score_sum = np.sum(scores, axis=axis, keepdims=True)

        # Avoid division by zero
        score_sum = np.maximum(score_sum, epsilon)

        return scores / score_sum

    def augment_lidar_class_scores_both(self, class_scores_r, class_scores_l, lidar_raw, projection_mats):
        """
        Projects lidar points onto segmentation map, appends class score each point projects onto.
        Then uses KNN to:
          (1) smooth low-confidence painted points,
          (2) propagate semantics to unpainted (missed) points.
        """
        #lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)
        # TODO: Project lidar points onto left and right segmentation maps. How to use projection_mats? 
        ################################
        lidar_cam_coords = self.cam_to_lidar(lidar_raw, projection_mats)

        # right
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_r = projection_mats['P3'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_r = points_projected_on_mask_r.transpose()
        points_projected_on_mask_r = points_projected_on_mask_r/(points_projected_on_mask_r[:,2].reshape(-1,1))

        true_where_x_on_img_r = (0 < points_projected_on_mask_r[:, 0]) & (points_projected_on_mask_r[:, 0] < class_scores_r.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_r = (0 < points_projected_on_mask_r[:, 1]) & (points_projected_on_mask_r[:, 1] < class_scores_r.shape[0])
        true_where_point_on_img_r = true_where_x_on_img_r & true_where_y_on_img_r

        points_projected_on_mask_r = points_projected_on_mask_r[true_where_point_on_img_r] # filter out points that don't project to image
        points_projected_on_mask_r = np.floor(points_projected_on_mask_r).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_r = points_projected_on_mask_r[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        # left
        lidar_cam_coords[:, -1] = 1 #homogenous coords for projection
        # TODO: change projection_mats['P2'] and projection_mats['R0_rect'] to be?
        points_projected_on_mask_l = projection_mats['P2'].dot(projection_mats['R0_rect'].dot(lidar_cam_coords.transpose()))
        points_projected_on_mask_l = points_projected_on_mask_l.transpose()
        points_projected_on_mask_l = points_projected_on_mask_l/(points_projected_on_mask_l[:,2].reshape(-1,1))

        true_where_x_on_img_l = (0 < points_projected_on_mask_l[:, 0]) & (points_projected_on_mask_l[:, 0] < class_scores_l.shape[1]) #x in img coords is cols of img
        true_where_y_on_img_l = (0 < points_projected_on_mask_l[:, 1]) & (points_projected_on_mask_l[:, 1] < class_scores_l.shape[0])
        true_where_point_on_img_l = true_where_x_on_img_l & true_where_y_on_img_l

        points_projected_on_mask_l = points_projected_on_mask_l[true_where_point_on_img_l] # filter out points that don't project to image
        points_projected_on_mask_l = np.floor(points_projected_on_mask_l).astype(int) # using floor so you don't end up indexing num_rows+1th row or col
        points_projected_on_mask_l = points_projected_on_mask_l[:, :2] #drops homogenous coord 1 from every point, giving (N_pts, 2) int array

        true_where_point_on_both_img = true_where_point_on_img_l & true_where_point_on_img_r
        true_where_point_on_img = true_where_point_on_img_l | true_where_point_on_img_r

        #indexing oreder below is 1 then 0 because points_projected_on_mask is x,y in image coords which is cols, rows while class_score shape is (rows, cols)
        #socre dimesion: point_scores.shape[2] TODO!!!!
        point_scores_r = class_scores_r[points_projected_on_mask_r[:, 1], points_projected_on_mask_r[:, 0]].reshape(-1, class_scores_r.shape[2])
        point_scores_l = class_scores_l[points_projected_on_mask_l[:, 1], points_projected_on_mask_l[:, 0]].reshape(-1, class_scores_l.shape[2])
        #augmented_lidar = np.concatenate((lidar_raw[true_where_point_on_img], point_scores), axis=1)
        augmented_lidar = np.concatenate((lidar_raw, np.zeros((lidar_raw.shape[0], class_scores_r.shape[2]))), axis=1)
        augmented_lidar[true_where_point_on_img_r, -class_scores_r.shape[2]:] += point_scores_r
        augmented_lidar[true_where_point_on_img_l, -class_scores_l.shape[2]:] += point_scores_l
        augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:] = 0.5 * augmented_lidar[true_where_point_on_both_img, -class_scores_r.shape[2]:]
        augmented_lidar = augmented_lidar[true_where_point_on_img]
        
        # === KNN SMOOTHING START ===
        # Only apply KNN if we have semantic scores and enough points
        if augmented_lidar.shape[0] > 0:
            from sklearn.neighbors import NearestNeighbors

            num_classes = class_scores_r.shape[2]
            xyz = augmented_lidar[:, :3]  # (N, 3)
            semantic_scores = augmented_lidar[:, -num_classes:]  # (N, C)

            # Compute max confidence per point
            max_conf = np.max(semantic_scores, axis=1)

            # Parameters for KNN smoothing
            K = 10
            CONF_THRESH = 1.0
            DIST_THRESH = 0.1  # meters

#             # Identify low-confidence points
#             low_conf_mask = max_conf < CONF_THRESH

#             if np.any(low_conf_mask):
#                 # Build KNN tree on all points
#                 nbrs = NearestNeighbors(n_neighbors=min(K, xyz.shape[0]), algorithm='kd_tree').fit(xyz)
#                 distances, indices = nbrs.kneighbors(xyz)  # (N, K)

#                 # Smooth only low-confidence points
#                 for i in np.where(low_conf_mask)[0]:
#                     # Filter neighbors within distance threshold
#                     valid_nbrs = distances[i] < DIST_THRESH
#                     if np.sum(valid_nbrs) <= 1:  # skip if only self
#                         continue
#                     nbr_indices = indices[i][valid_nbrs]
#                     # Use mean of neighbors' semantic scores
#                     smoothed_score = np.mean(semantic_scores[nbr_indices], axis=0)
#                     augmented_lidar[i, -num_classes:] = smoothed_score
                    
#             # Identify low-confidence points
#             low_conf_mask = max_conf < CONF_THRESH

#             if np.any(low_conf_mask):
#                 # Build KNN tree on all points
#                 nbrs = NearestNeighbors(n_neighbors=min(K, xyz.shape[0]), algorithm='kd_tree').fit(xyz)
#                 distances, indices = nbrs.kneighbors(xyz)  # (N, K)

#                 # Smooth only low-confidence points
#                 for i in np.where(low_conf_mask)[0]:
#                     # Filter neighbors within distance threshold
#                     valid_nbrs = distances[i] < DIST_THRESH
#                     if np.sum(valid_nbrs) <= 1:  # skip if only self
#                         continue
#                     nbr_indices = indices[i][valid_nbrs]
#                     # Use mean of neighbors' semantic scores
#                     smoothed_score = np.mean(semantic_scores[nbr_indices], axis=0)
#                     augmented_lidar[i, -num_classes:] = max_conf[i] * semantic_scores[i] + (1 - max_conf[i]) * smoothed_score
                    
        low_conf_mask = max_conf < CONF_THRESH
        
        # parallelized
        if np.any(low_conf_mask):
            # Build KNN tree once
            nbrs = NearestNeighbors(n_neighbors=min(K, xyz.shape[0]), algorithm='kd_tree').fit(xyz)
            distances, indices = nbrs.kneighbors(xyz)  # (N, K)

            # Get indices of low-confidence points
            low_idx = np.where(low_conf_mask)[0]  # shape: (M,)

            # For these points, get their neighbor distances and indices
            nbr_dists = distances[low_idx]        # (M, K)
            nbr_inds = indices[low_idx]           # (M, K)

            # Mask neighbors within distance threshold AND not self-only
            valid_nbr_mask = nbr_dists < DIST_THRESH  # (M, K)
            # Optional: exclude self if needed (usually index 0 is self in k=1, but not guaranteed)
            # Here we rely on DIST_THRESH to filter out far neighbors

            # Count valid neighbors per point
            num_valid = np.sum(valid_nbr_mask, axis=1)  # (M,)
            has_enough_nbrs = num_valid > 1

            if np.any(has_enough_nbrs):
                # Select only points with enough neighbors
                process_mask = has_enough_nbrs
                process_idx = low_idx[process_mask]               # (P,)
                process_nbr_inds = nbr_inds[process_mask]         # (P, K)
                process_valid_mask = valid_nbr_mask[process_mask] # (P, K)

                # Flatten neighbor indices for advanced indexing
                # We'll gather semantic scores for all valid neighbors
                row_indices = np.repeat(np.arange(process_nbr_inds.shape[0]), process_nbr_inds.shape[1])
                col_indices = process_nbr_inds.ravel()
                valid_flat = process_valid_mask.ravel()

                # Gather all neighbor semantic scores
                all_nbr_scores = semantic_scores[col_indices]  # (P*K, C)
                all_nbr_scores = all_nbr_scores[valid_flat]    # (Q, C), Q = total valid neighbors

                # Compute start/end for each point's neighbors
                counts = np.sum(process_valid_mask, axis=1)    # (P,)
                starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

                # Compute mean per point
                smoothed_scores = np.zeros((len(counts), semantic_scores.shape[1]), dtype=np.float32)
                for j in range(len(counts)):
                    if counts[j] > 0:
                        smoothed_scores[j] = np.mean(all_nbr_scores[starts[j]:starts[j]+counts[j]], axis=0)

                # Confidence-weighted fusion
                orig_scores = semantic_scores[process_idx]      # (P, C)
                conf_weights = max_conf[process_idx][:, None]   # (P, 1)
                fused_scores = conf_weights * orig_scores + (1 - conf_weights) * smoothed_scores

                # Write back to augmented_lidar
                augmented_lidar[process_idx, -num_classes:] = fused_scores
                # augmented_lidar[process_idx, -num_classes:] = self.l1_normalize(fused_scores)
        # === KNN SMOOTHING END ===
        
        augmented_lidar = self.create_cyclist(augmented_lidar)

        return augmented_lidar

    def run(self):
#         # num_image = 7481
#         num_image = 2000
#         for idx in tqdm(range(num_image)):
#             sample_idx = "%06d" % idx
#             # points: N * 4(x, y, z, r)
#             points = self.get_lidar(sample_idx)
            
#             # get segmentation score from network
#             scores_from_cam = self.get_score(sample_idx, "image_2/")
#             scores_from_cam_r = self.get_score(sample_idx, "image_3/")
#             # scores_from_cam: H * W * 4/5, each pixel have 4/5 scores(0: background, 1: bicycle, 2: car, 3: person, 4: rider)
            
#             # print(scores_from_cam.shape)
#             # print(scores_from_cam[0])
#             # print(scores_from_cam_r.shape)

#             # get calibration data
#             calib_fromfile = self.get_calib_fromfile(sample_idx)
            
#             # paint the point clouds
#             # points: N * 8
#             points = self.augment_lidar_class_scores_both(scores_from_cam_r, scores_from_cam, points, calib_fromfile)
            
#             np.save(self.save_path + ("%06d.npy" % idx), points)

            idx = 8
            sample_idx = "%06d" % idx
            # points: N * 4(x, y, z, r)
            points = self.get_lidar(sample_idx)
            
            # get segmentation score from network
            scores_from_cam = self.get_score(sample_idx, "image_2/")
            scores_from_cam_r = self.get_score(sample_idx, "image_3/")
            # scores_from_cam: H * W * 4/5, each pixel have 4/5 scores(0: background, 1: bicycle, 2: car, 3: person, 4: rider)
            
            # print(scores_from_cam.shape)
            # print(scores_from_cam[0])
            # print(scores_from_cam_r.shape)

            # get calibration data
            calib_fromfile = self.get_calib_fromfile(sample_idx)
            
            # paint the point clouds
            # points: N * 8
            points = self.augment_lidar_class_scores_both(scores_from_cam_r, scores_from_cam, points, calib_fromfile)
            
            # np.save(self.save_path + ("%06d.npy" % idx), points)
            points.astype(np.float32).tofile(self.save_path + ("%06d.bin" % idx))

if __name__ == '__main__':
    painter = Painter(SEG_NET)
    painter.run()