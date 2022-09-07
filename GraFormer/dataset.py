import numpy as np
import cv2
import io
import os
import torch.utils.data as data
import pickle
import re
from PIL import Image

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', seq_length=1, n_points=29, eval_pred=False):

        self.root = root 
        self.load_set = load_set  # 'train','val','test'
        self.seq_length = seq_length
        self.n_points = n_points
        self.eval_pred = eval_pred
        if eval_pred:
            self.rcnn_dict = pickle.load(open(f'./rcnn_outputs/rcnn_outputs_{self.n_points}_{self.load_set}.pkl', 'rb'))
            # print(len(self.rcnn_dict.keys()))
        # print(self.rcnn_dict)
        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        self.count = 0

        if self.n_points > 29:
            self.mesh2d = np.load(os.path.join(root, 'mesh2d-%s.npy'%self.load_set))[:, :778]
            self.mesh3d = np.load(os.path.join(root, 'mesh3d-%s.npy'%self.load_set))[:, :778]
            # print(self.mesh2d.shape)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        
        if self.n_points > 29: # i.e. mesh
            point2d = self.mesh2d[index][:778]
            point3d = self.mesh3d[index][:778]

        else:    
            point2d = self.points2d[index][:self.n_points]
            point3d = self.points3d[index][:self.n_points]

        if self.eval_pred:
            path = self.images[index]
            point2d = self.rcnn_dict[path][:, :2]

        if self.seq_length > 1:
            path = self.images[index]
            frame_num = int(path.split('/')[-1].split('.')[0])
            point2d, point3d = self.create_sequence(index, frame_num, path)
        
        return point2d, point3d

    def __len__(self):
        return len(self.images)

    def create_sequence(self, index, frame_num, path):

        point2d_seq = np.zeros((self.seq_length, self.n_points, 2))
        point3d_seq = np.zeros((self.seq_length, self.n_points, 3))
        
        missing_frames = False
        
        for i in range(0, self.seq_length):
            if frame_num - i < 0:
                missing_frames = True
                break
            new_frame_num = '%04d' % (frame_num-i)
            new_path = re.sub('\d{4}', new_frame_num, path)    
            if self.eval_pred and new_path in self.rcnn_dict.keys():
                point2d_seq[-i-1] = self.rcnn_dict[new_path][:self.n_points, :2] 
                point3d_seq[-i-1] = self.points3d[index-i][:self.n_points]
                last_pose = -i-1
            elif new_path == self.images[index-i]: # Check that the next sample in the list is the temporally adjacent frame
                point2d_seq[-i-1] = self.points2d[index-i][:self.n_points]
                point3d_seq[-i-1] = self.points3d[index-i][:self.n_points]
                last_pose = -i-1
            else: # Replicate the last pose in case of missing information
                point2d_seq[-i-1] = point2d_seq[last_pose]
                point3d_seq[-i-1] = point3d_seq[last_pose]

            
            
        if missing_frames:
            n_missing_frames = self.seq_length - i

            point2d_seq[0:-i] = np.tile(point2d_seq[-i], (n_missing_frames, 1, 1)) 
            point3d_seq[0:-i] = np.tile(point3d_seq[-i], (n_missing_frames, 1, 1))
        
        point2d_seq = point2d_seq.reshape((self.seq_length * self.n_points, 2))
        point3d_seq = point3d_seq.reshape((self.seq_length * self.n_points, 3))

        return point2d_seq, point3d_seq

