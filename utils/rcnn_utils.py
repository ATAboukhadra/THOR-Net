import torch
import numpy as np

def calculate_bounding_box(point2d, increase=False):
    pad_size = 15
    x_min = int(min(point2d[:,0]))
    y_min = int(min(point2d[:,1]))
    x_max = int(max(point2d[:,0]))
    y_max = int(max(point2d[:,1]))
    
    if increase:
        return np.array([x_min - pad_size, y_min - pad_size, x_max + pad_size, y_max + pad_size])
    else:
        return np.array([x_min, y_min, x_max, y_max])


def create_rcnn_data(bb, point2d, point3d, num_keypoints=21):
    ''' Prepares data for an RCNN by creating tensors for Bounding boxes, labels and keypoints with their visibility'''
            
    # Boxes and Labels
    boxes = torch.Tensor(bb[np.newaxis, ...]).float()
    labels = torch.from_numpy(np.array([1])) # 1 for hand-object box or hand-only box
    
    # Appending visibility TODO: change this to actual visibility
    visibility = np.ones(num_keypoints).reshape(-1, 1)
    keypoints = np.append(point2d[:num_keypoints], visibility, axis=1)

    # Append keypoints
    final_keypoints = torch.Tensor(keypoints[:num_keypoints][np.newaxis, ...]).float()
    final_keypoints3d = torch.Tensor(point3d[:num_keypoints][np.newaxis, ...]).float()

    return boxes, labels, final_keypoints, final_keypoints3d
