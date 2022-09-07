import torch
import torch.nn.functional as F
import os
from torchvision.ops import roi_align
from typing import Optional, List, Dict, Tuple
from torch import nn, Tensor

# from pytorch3d.loss import (
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
#     chamfer_distance
# )
# from pytorch3d.structures import Meshes
# from pytorch3d.io import load_obj, save_obj

def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid

def heatmaps_to_keypoints(maps, rois):
    """Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = widths.clamp(min=1)
    heights = heights.clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_keypoints = maps.shape[1]

    xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
    end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
    for i in range(len(rois)):
        roi_map_width = int(widths_ceil[i].item())
        roi_map_height = int(heights_ceil[i].item())
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = F.interpolate(
            maps[i][:, None], size=(roi_map_height, roi_map_width), mode='bicubic', align_corners=False)[:, 0]
        # roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)

        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode='floor')
        # assert (roi_map_probs[k, y_int, x_int] ==
        #         roi_map_probs[k, :, :].max())
        x = (x_int.float() + 0.5) * width_correction
        y = (y_int.float() + 0.5) * height_correction
        xy_preds[i, 0, :] = x + offset_x[i]
        xy_preds[i, 1, :] = y + offset_y[i]
        xy_preds[i, 2, :] = 1
        end_scores[i, :] = roi_map[torch.arange(num_keypoints, device=roi_map.device), y_int, x_int]

    return xy_preds.permute(0, 2, 1), end_scores

def keypointrcnn_loss(keypoint_logits, proposals, gt_keypoints, keypoint_matched_idxs, 
                    keypoint3d_pred=None, keypoint3d_gt=None, mesh3d_pred=None, 
                    mesh3d_gt=None, original_images=None, palms_gt=None,
                    photometric=False, num_classes=4):

    N, K, H, W = keypoint_logits.shape
    assert H == W
    discretization_size = H
    heatmaps = []
    valid = []
    kps3d = []
    meshes3d = []
    images = []
    palms = []
    
    if palms_gt is None:
        palms_gt = [None] * len(proposals)

    zipped_data = zip(proposals, gt_keypoints, keypoint3d_gt, mesh3d_gt, original_images, keypoint_matched_idxs, palms_gt)
    # zipped_data = zip(proposals, gt_keypoints, keypoint_matched_idxs, original_images)
    for proposals_per_image, gt_kp_in_image, gt_kp3d_in_image, gt_mesh3d_in_image, image, midx, palm_in_image in zipped_data:
        
        kp = gt_kp_in_image[midx]
        
        if palm_in_image is not None:
            palm = palm_in_image[midx]
            palms.append(palm.view(-1))

        num_regions = midx.shape[0]

        heatmaps_per_image, valid_per_image = keypoints_to_heatmap(kp, proposals_per_image, discretization_size)
        
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

        if num_classes == 2:
            kp3d = gt_kp3d_in_image[midx]
            mesh3d = gt_mesh3d_in_image[midx]
            kps3d.append(kp3d.view(-1))
            meshes3d.append(mesh3d.view(-1))
            images.extend([image] * num_regions)
        else:
            images.append(image)
        
    keypoint_targets = torch.cat(heatmaps, dim=0)
    valid = torch.cat(valid, dim=0).to(dtype=torch.uint8)    
    valid = torch.where(valid)[0]
    
    # torch.mean (in binary_cross_entropy_with_logits) does'nt
    # accept empty tensors, so handle it sepaartely
    
    if keypoint_targets.numel() == 0 or len(valid) == 0:
        return keypoint_logits.sum() * 0

    keypoint_logits = keypoint_logits.view(N * K, H * W)
    
    # Heatmap Loss
    keypoint_loss = F.cross_entropy(keypoint_logits[valid], keypoint_targets[valid])
    
    # 3D pose Loss
    if num_classes > 2:
        keypoint3d_targets = torch.cat(keypoint3d_gt, dim=0)
        mesh3d_targets = torch.cat(mesh3d_gt, dim=0) 

    else:
        palms = torch.cat(palms, dim=0).view(N, 1, 3) # TODO: add another condition for this
        keypoint3d_targets = torch.cat(kps3d, dim=0).view(N, K, 3)
        mesh3d_targets = torch.cat(meshes3d, dim=0)

    N, K, D = keypoint3d_targets.shape
    keypoint3d_pred = keypoint3d_pred.view(N * K, 3)
    keypoint3d_targets = keypoint3d_targets.view(N * K, 3)
    keypoint3d_loss = F.mse_loss(keypoint3d_pred, keypoint3d_targets) / 1000
    
    # 3D shape Loss
    N, K, D = mesh3d_pred[:, :, :3].shape
    xyz_rgb_pred = torch.clone(mesh3d_pred)
    mesh3d_pred = torch.reshape(mesh3d_pred[:, :, :3], (N * K, D)) 
    mesh3d_targets = torch.reshape(mesh3d_targets, (N * K, D))
    mesh3d_loss = F.mse_loss(mesh3d_pred, mesh3d_targets) / 1000

    # Photometric Loss
    # To penalize rgb values with the projection of the GT shape, replace predicted xyz with GT xyz
    mesh3d_targets = torch.reshape(mesh3d_targets, (N, K, D))    
    
    # Calculate photometric loss
    if photometric:
        pred_rgb = xyz_rgb_pred[:, :, 3:]
        pts3D = mesh3d_targets
        # pts3D = xyz_rgb_pred[:, :, :3]
        images = torch.stack(images)
        photometric_loss = calculate_photometric_loss(pts3D, pred_rgb, images, N, K)
    else:
        photometric_loss = None
    # mesh3d_loss_smooth = calculate_smoothing_loss(mesh3d_pred[:K], K)
    # if N > 1:
    #     mesh3d_loss_smooth += calculate_smoothing_loss(mesh3d_pred[K:K*2], K)
    # mesh3d_loss += mesh3d_loss_smooth
    
    # Print the losses
    return keypoint_loss, keypoint3d_loss, mesh3d_loss, photometric_loss

def keypointrcnn_inference(x, boxes):
    # type: (Tensor, List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    kp_probs = []
    kp_scores = []

    boxes_per_image = [box.size(0) for box in boxes]
    x2 = x.split(boxes_per_image, dim=0)

    for xx, bb in zip(x2, boxes):
        kp_prob, scores = heatmaps_to_keypoints(xx, bb)
        kp_probs.append(kp_prob)
        kp_scores.append(scores)

    return kp_probs, kp_scores

def get_hand_object_faces(kps=778):
    src_obj = os.path.join('../HOPE/datasets/hands', 'hand_model_778.obj')
    verts, faces, aux = load_obj(src_obj)
    hand_faces = faces.verts_idx

    if kps > 778:
        src_obj = os.path.join('../HOPE/datasets/spheres', 'sphere_1000.obj')
        verts, faces, aux = load_obj(src_obj)
        object_faces = faces.verts_idx
        object_faces = object_faces + 778

        final_faces = torch.cat((hand_faces, object_faces), axis=0)
    
    else:
        final_faces = hand_faces
    return final_faces

def calculate_smoothing_loss(mesh3d, K=778):
    
    faces_idx = get_hand_object_faces(K).to('cuda:1')
            
    trg_mesh = Meshes(verts=[mesh3d], faces=[faces_idx])
            
    # src_mesh = Meshes(verts=[keypoint_targets3d[valid]], faces=[faces_idx])
    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    # loss_chamfer, _ = chamfer_distance(keypoint3d_pred[valid], keypoint_targets3d[valid])
    
    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(trg_mesh)
    # mesh normal consistency
    loss_normal = mesh_normal_consistency(trg_mesh)
    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(trg_mesh, method="uniform")
    # Weighted sum of the losses
    # print(loss_edge, loss_laplacian* 0.1 , loss_normal * 0.01)
    mesh3d_loss_smooth = 0.01 * loss_edge + loss_laplacian * 0.001  + loss_normal * 0.00001

    return mesh3d_loss_smooth

def append_rois_shapes(keypoint_proposals, image_shapes, kps, scale):
    rois_with_shapes = []

    for i, p in enumerate(keypoint_proposals):
        n_rois = p.shape[0]
        img_shape = torch.Tensor(image_shapes[i]).unsqueeze(axis=0).repeat(n_rois, 1).to(p.device)
        roi_with_shape = torch.cat((p, img_shape), axis=1)
        rois_with_shapes.append(roi_with_shape)

    rois_tensor = torch.cat(rois_with_shapes, dim=0).unsqueeze(axis=1)
    rois_tensor = rois_tensor.repeat(1, kps, 1) / scale
    
    return rois_tensor

def filter_rois(keypoint_proposals, training, labels=None):
    new_keypoint_proposals = []

    if training:
        for i in range(len(keypoint_proposals)):
            new_keypoint_proposals.append(keypoint_proposals[i][-3:])

    else:
        for i in range(len(keypoint_proposals)):
            
            labels_list = labels[i].tolist()
            if not set([1, 2, 3]).issubset(labels_list):
                return None
            lh_roi = keypoint_proposals[i][labels_list.index(1)]
            rh_roi = keypoint_proposals[i][labels_list.index(2)]
            obj_roi = keypoint_proposals[i][labels_list.index(3)]
            rois = torch.stack([lh_roi, rh_roi, obj_roi], dim=0)
            new_keypoint_proposals.append(rois)

    return new_keypoint_proposals

def project_3D_points(pts3D):

    # H2O matrix

    cam_mat = torch.Tensor(
        [[636.6593, 0       , 635.2839],
        [0        , 636.2520, 366.8740],
        [0        , 0       , 1]]).to(pts3D.device)
    
    # HO3D matrix

    # cam_mat = torch.Tensor(
    #     [[617.343,0,      312.42],
    #     [0,       617.343,241.42],
    #     [0,       0,       1]]).to(pts3D.device)

    K = pts3D.shape[0]
    pts2D = torch.zeros((K, 2)).to(torch.long)

    proj_pts = pts3D.matmul(cam_mat.T)
    if torch.all(proj_pts[:, 2:]):
        pts2D = torch.stack([proj_pts[:,0] / proj_pts[:,2], proj_pts[:,1] / proj_pts[:,2]], axis=1).to(torch.long)
    return pts2D

def calculate_photometric_loss(pts3D, rgb, images, N, K, centers=None):

    if centers is not None:
        pts3D = pts3D + centers
    
    pts3D = pts3D.reshape((N * K, 3))

    pts2D = project_3D_points(pts3D)
    pts2D = pts2D.view(N, K, 2)
    B, H, W, _ = images.shape

    idx_x = pts2D[:, :, 0].clamp(min=0, max=W-1)
    idx_y = pts2D[:, :, 1].clamp(min=0, max=H-1)
    
    pixels = images[torch.arange(B).unsqueeze(1), idx_y, idx_x]

    photometric_loss = F.mse_loss(rgb, pixels)

    return photometric_loss
