from __future__ import absolute_import, division

import numpy as np
import torch
from .camera import world_to_camera, normalize_screen_coordinates


def create_2d_data(data_path, dataset):
    keypoints = np.load(data_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    return keypoints


def read_3d_data(dataset):
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]

            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, :] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

    return dataset


def fetch(subjects, dataset, keypoints, action_filter=None, stride=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_actions = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    # if action.startswith(a):
                    if action.split(' ')[0] == a:
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                out_actions.append([action.split(' ')[0]] * poses_2d[i].shape[0])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_poses_3d) == 0:
        out_poses_3d = None

    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            out_actions[i] = out_actions[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d, out_actions


def convert_faces_to_edges(faces):
    edges = []
    for face in faces:
        edges.append([face[0], face[1]])
        edges.append([face[1], face[2]])
        edges.append([face[2], face[0]])
    
    return edges

def create_edges(seq_length=1, num_nodes=29):

    if num_nodes == 778:
        
        faces = np.load('./GraFormer/RightHandFaces.npy')
        edges = convert_faces_to_edges(faces)
        edges = torch.tensor(edges, dtype=torch.long)
        return edges
    else:
        initial_edges = [
                    # Hand connectivity
                    [0, 1], [1, 2], [2, 3], [3, 4], 
                    [0, 5], [5, 6], [6, 7], [7, 8], 
                    [0, 9], [9, 10], [10, 11], [11, 12],
                    [0, 13], [13, 14], [14, 15], [15, 16], 
                    [0, 17], [17, 18], [18, 19], [19, 20]]
        if num_nodes == 29:
                    # Object connectivity
                    initial_edges.extend([
                    [21, 22],[22, 24], [24, 23], [23, 21],
                    [25, 26], [26, 28], [28, 27], [27, 25],
                    [21, 25], [22, 26], [23, 27], [24, 28]])
    edges = []
    for i in range(0, seq_length):

        # Create a translated version of edges to be spatial edges of the next temporal pose
        spatial_edges = (np.array(initial_edges) + i * num_nodes).tolist()
        edges.extend(spatial_edges)
        
        # Connect every node with its corresponsing node in the temporally-adjacent pose
        if i > 0:
            start = (i-1) * num_nodes
            end = i * num_nodes
            temporal_edges = [[i, i + num_nodes] for i in range(start, end)]
            edges.extend(temporal_edges)

    edges = torch.tensor(edges, dtype=torch.long)
    return edges