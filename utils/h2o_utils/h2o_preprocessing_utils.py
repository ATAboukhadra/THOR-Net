
import os
import tarfile

import numpy as np
import torch

from .h2o_annotation_decoding import decode_cam_intrinsics
from manopth.manolayer import ManoLayer
from torch.utils.data.datapipes.datapipe import DataChunk


class Minimal(object):
    def __init__(self, **kwargs):
        self.__dict__ = kwargs


def read_obj(filename):
    """ Reads the Obj file. Function reused from Matthew Loper's OpenDR package"""

    lines = open(filename).read().split('\n')

    d = {'v': [], 'f': []}

    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0]) - 1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v', 'f']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result


class Preprocessor:
    def __init__(self, mano_model_path, objects_path, ds_dir):
        self.lh_model = ManoLayer(mano_root=mano_model_path, use_pca=False, ncomps=45, flat_hand_mean=True, side='left')
        self.rh_model = ManoLayer(mano_root=mano_model_path, use_pca=False, ncomps=45, flat_hand_mean=True,
                                  side='right')

        self.objects_verts_dict, self.objects_faces_dict = {}, {}
        self.load_object_files(objects_path)
        self.object_map = {1: 'book', 2: 'espresso', 3: 'lotion', 4: 'lotion_spray',
                           5: 'milk', 6: 'cocoa', 7: 'chips', 8: 'cappuccino'}
        self.calibration_dict = {}
        self.load_cam_int(ds_dir)

    def decode_mano(self, mano_params, is_right):
        """ Uses MANO models to decode mano parameters """
        if mano_params[0] == 0:
            return torch.zeros((778, 3))

        transl = mano_params[1:4].view(1, -1)
        pose = mano_params[4:52].view(1, -1)
        betas = mano_params[52:62].view(1, -1)

        if is_right:
            model = self.rh_model
        else:
            model = self.lh_model

        output = model(pose, betas, transl)

        return output[0].reshape((778, 3))

    def load_object_files(self, objects_path: str):
        """ Initializes object dictionaries with all object meshes as vertices and faces """
        files = os.listdir(objects_path)

        for f in files:
            name = f.split('.')[0]

            object_mesh = read_obj(os.path.join(objects_path, f))

            object_mesh_vertices = object_mesh.v
            object_mesh_faces = object_mesh.f

            self.objects_verts_dict[name] = torch.Tensor(object_mesh_vertices)
            self.objects_faces_dict[name] = object_mesh_faces

    def load_object_mesh(self, label: int):
        """ Load object mesh based on categorical label """
        class_name = self.object_map[label]
        return self.objects_verts_dict[class_name]

    def load_cam_int(self, ds_dir: str):
        """ Initializes a dict that contains the camera intrinsics matrix for each combination of subject, scene,
         object, and camera.
        """
        tar = tarfile.open(os.path.join(ds_dir, 'camera_calibration.tar'))
        members = tar.getmembers()
        for member in members:
            f = tar.extractfile(member)
            key = member.name.split('.')[0]
            self.calibration_dict[key] = decode_cam_intrinsics(f.read())

    def project_3D_points(self, points3d, key):
        """ Project 3D points to 2D using the respective camera matrix"""
        K, D = points3d.shape
        points2d = torch.zeros((K, 2))
        cam_intr = self.calibration_dict[key]
        points_proj = cam_intr.matmul(points3d.t()).t()
        # Avoid division by zero when all points are at zero
        if torch.all(points_proj[:, 2:]):
            points2d = (points_proj / points_proj[:, 2:])[:, :2]
        return points2d


def calculate_bounding_box(point2d, increase=False):
    """ Compute bounding box of a group of 2D points"""
    pad_size = 15
    x_min = int(min(point2d[:, 0]))
    y_min = int(min(point2d[:, 1]))
    x_max = int(max(point2d[:, 0]))
    y_max = int(max(point2d[:, 1]))

    if increase:
        return np.array([x_min - pad_size, y_min - pad_size, x_max + pad_size, y_max + pad_size])
    else:
        return np.array([x_min, y_min, x_max, y_max])


def create_rcnn_data(point2d, merge_objects=False):
    """Prepares data for an R-CNN by creating tensors for bounding boxes, labels and keypoints with their visibility"""

    K, D = point2d.shape
    if merge_objects:
        bb = calculate_bounding_box(point2d, increase=True)
        # Boxes and Labels
        boxes = torch.Tensor(bb[np.newaxis, ...]).float()
        labels = torch.from_numpy(np.array([1]))  # 1 for hand-object box or hand-only box

        # Appending visibility 
        visibility = np.ones(K).reshape(-1, 1)
        keypoints = np.append(point2d, visibility, axis=1)

        # Append keypoints
        final_keypoints = torch.Tensor(keypoints[np.newaxis, ...]).float()

    else:
        bb_left = calculate_bounding_box(point2d[:21], increase=True)
        bb_right = calculate_bounding_box(point2d[21:42], increase=True)
        bb_object = calculate_bounding_box(point2d[42:], increase=True)

        # Boxes & Labels
        boxes = torch.Tensor(np.vstack((bb_left, bb_right, bb_object))).float()
        labels = torch.from_numpy(np.array([1, 2, 3]))  # 1 for left hand, 2 for right hand and 3 for object

        # Appending visibility 
        visibility = np.ones(point2d.shape[0]).reshape(-1, 1)
        keypoints = np.append(point2d, visibility, axis=1)
        left_keypoints = keypoints[:21]
        right_keypoints = keypoints[21:42]
        obj_keypoints = keypoints[42:]

        # Append dummy points to the objects keypoints to match the number of hands keypoints
        hand_obj_diff = left_keypoints.shape[0] - obj_keypoints.shape[0]
        dummy_obj_points = np.zeros((hand_obj_diff, 3))
        obj_keypoints = np.append(obj_keypoints, dummy_obj_points, 0)

        # Append keypoints
        final_keypoints = torch.Tensor(np.stack((left_keypoints, right_keypoints, obj_keypoints))).float()

    return boxes, labels, final_keypoints


def index_containing_substring(filenames, substring):
    # Helper function to find the index of a string that contains a substring in a list
    for i, s in enumerate(filenames):
        if substring in s:
            return i
    return -1


class MyPreprocessor:

    def __init__(self, mano_model_path: str, objects_path: str, ds_dir: str):
        self.preprocessor = Preprocessor(mano_model_path, objects_path, ds_dir)

    def __call__(self, sample: DataChunk):
        """ This is an example of how to preprocess the annotations and create a new tuple.

        :param sample: The sample that shall be preprocessed.
        :return: The new sample.
        """

        # Get the index of the required data
        filenames = [s[0] for s in sample]
        data = ['rgb', 'hand_pose', 'mano', 'obj_pose', 'obj_pose_rt']
        data_idx = [index_containing_substring(filenames, s) for s in data]
        data_idx_dict = dict(zip(data, data_idx))

        # RGB image is the first element in the tuple
        rgb_path = sample[data_idx_dict['rgb']][0]
        rgb_image = sample[data_idx_dict['rgb']][1]

        # Subject / Scene / object / cam key for the sample
        key = sample[0][0].split('/')[-1].split('.')[0][:-7]

        # Hand pose
        hand_pose3d = sample[data_idx_dict['hand_pose']][1]

        # Decoding Mano Params
        mano_params = sample[data_idx_dict['mano']][1]
        left_hand_verts = self.preprocessor.decode_mano(mano_params[:62], is_right=False)
        right_hand_verts = self.preprocessor.decode_mano(mano_params[62:], is_right=True)

        # Object label is stored at the first index of the last row of the object pose
        object_pose = sample[data_idx_dict['obj_pose']][1]
        object_label = int(object_pose[-1][0])
        # Load Object Mesh
        object_verts = self.preprocessor.load_object_mesh(object_label)

        # Project object vertices to camera coordinates
        obj_pose_rt = sample[data_idx_dict['obj_pose_rt']][1]
        homogeneous_ones = torch.ones((object_verts.shape[0], 1))
        object_verts = torch.cat((object_verts, homogeneous_ones), axis=1)
        object_verts = obj_pose_rt.matmul(object_verts.T).T[:, :-1] * 1000

        # Append hand meshes with object mesh
        mesh3d = torch.cat((left_hand_verts, right_hand_verts, object_verts), axis=0)

        # Append hand poses with object pose
        pose3d = torch.cat((hand_pose3d, object_pose[:-1]), axis=0)

        # Project 3D pose and 3D shape into 2D using camera intrinsics
        mesh2d = self.preprocessor.project_3D_points(mesh3d, key)
        pose2d = self.preprocessor.project_3D_points(pose3d, key)

        # Create bounding boxes and keypoints to train Keypoint RCNN
        boxes, labels, keypoints = create_rcnn_data(pose2d, merge_objects=False)

        # Return the new sample as a dict
        new_sample = (rgb_path, rgb_image, pose2d, pose3d, mesh2d, mesh3d, boxes, labels, keypoints)

        output_dict = {
            'rgb_path': rgb_path,
            'rgb_image': rgb_image,
            'pose2d': pose2d,
            'pose3d': pose3d,
            'mesh2d': mesh2d,
            'mesh3d': mesh3d,
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        return new_sample