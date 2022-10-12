import os
import numpy as np
import trimesh
import pickle
import torch
import cv2
import os
import argparse
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from manopth.manolayer import ManoLayer
from sklearn.preprocessing import MinMaxScaler
import joblib

# Change this path

# Input parameters
parser = argparse.ArgumentParser()

# Loading dataset    
parser.add_argument("--root", required=True, help="HO3D dataset folder")
parser.add_argument("--mano_root", required=True, help="Path to MANO models")
parser.add_argument("--YCBModelsDir", default='./datasets/ycb_models', help="Path to YCB object meshes folder")
parser.add_argument("--dataset_path", default='./datasets/ho3d', help="Where to store dataset files")

args = parser.parse_args()

root = args.root
YCBModelsDir = args.YCBModelsDir
dataset_path = args.dataset_path
mano_root = args.mano_root

evaluation = os.path.join(root, 'evaluation')
train = os.path.join(root, 'train')


# Load object mesh
reorder_idx = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])

coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)

def fit_scaler(arr, k):

    scaler = MinMaxScaler()
    scaled = scaler.fit(arr)
    print(f'{k} scaler min:', scaler.data_min_, ', scaler max:', scaler.data_max_)
    joblib.dump(scaler, f'{k}_scaler.save') 

    return scaler

def normalize(arr, normalizer):

    all_points = arr.reshape((-1, arr.shape[-1]))
    normalized_points = normalizer.transform(all_points)

    return normalized_points.reshape(arr.shape)

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
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])

    for k, v in d.items():
        if k in ['v','f']:
            if v:
                d[k] = np.vstack(v)
            else:
                print(k)
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result

obj_dict = {}
def load_ycb_obj(ycb_dataset_path, obj_name, rot=None, trans=None, decimationValue=1000):
    ''' Load a YCB mesh based on the name '''
    if obj_name not in obj_dict.keys():
        path = os.path.join(ycb_dataset_path, obj_name, f'morphed_sphere_{decimationValue}.obj')
        obj_mesh = read_obj(path)
        obj_dict[obj_name] = obj_mesh        
    else:
        obj_mesh = obj_dict[obj_name]
    # apply current pose to the object model
    if rot is not None:
        obj_mesh_verts = np.matmul(obj_mesh.v, cv2.Rodrigues(rot)[0].T) + trans

    # Change to non openGL coords and convert from m to mm
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    obj_mesh_verts = obj_mesh_verts.dot(coordChangeMat.T) * 1000
        
    return obj_mesh_verts, obj_mesh.f

def project_3D_points(cam_mat, pts3D, is_OpenGL_coords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2

    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    if is_OpenGL_coords:
        pts3D = pts3D.dot(coord_change_mat.T)

    proj_pts = pts3D.dot(cam_mat.T)
    proj_pts = np.stack([proj_pts[:,0]/proj_pts[:,2], proj_pts[:,1]/proj_pts[:,2]],axis=1)

    assert len(proj_pts.shape) == 2

    return proj_pts

def load_mesh_from_manolayer(fullpose, beta, trans, mano_layer):
    
    # Convert inputs to tensors and reshape them to be compatible with Mano Layer
    fullpose_tensor = torch.as_tensor(fullpose, dtype=torch.float32).reshape(1, -1)
    shape_tensor = torch.as_tensor(beta, dtype=torch.float32).reshape(1, -1)
    trans_tensor = torch.as_tensor(trans, dtype=torch.float32).reshape(1, -1)

    # Pass to Mano layer
    hand_verts, hand_joints = mano_layer(fullpose_tensor, shape_tensor, trans_tensor)
    
    # return outputs as numpy arrays and scale them back from mm to m 
    hand_verts = hand_verts.cpu().detach().numpy()[0] 
    hand_joints = hand_joints.cpu().detach().numpy()[0]

    return hand_joints, hand_verts, mano_layer.th_faces


def load_annotations(data, mano_layer, subset='train'):

    cam_intr = data['camMat']

    if subset == 'train':
        hand3d = data['handJoints3D'][reorder_idx]
    else:
        hand3d = data['handJoints3D'].reshape((1, -1))

    obj_corners = data['objCorners3D']
    # print(data)
    # print(len(data['handBoundingBox']))
    # Convert to non-OpenGL coordinates and multiply by thousand to convert from m to mm
    hand_object3d = np.concatenate([hand3d, obj_corners]) * 1000
    # hand_object3d = hand_object3d.dot(coordChangeMat.T)

    # Project from 3D world to Camera coordinates using the camera matrix  
    hand_object3d = hand_object3d.dot(coordChangeMat.T)
    hand_object_proj = cam_intr.dot(hand_object3d.transpose()).transpose()
    hand_object2d = (hand_object_proj / hand_object_proj[:, 2:])[:, :2]

    mesh3d = np.array([])
    mesh2d = np.array([])
    if subset == 'train':
        _, hand_mesh3d, _ = load_mesh_from_manolayer(data['handPose'], data['handBeta'], data['handTrans'], mano_layer)
        
        # Project from 3D world to Camera coordinates using the camera matrix    
        hand_mesh3d = hand_mesh3d.dot(coordChangeMat.T)
        hand_mesh_proj = cam_intr.dot(hand_mesh3d.transpose()).transpose()
        hand_mesh2d = (hand_mesh_proj / hand_mesh_proj[:, 2:])[:, :2]

        # Do the same for the object
        obj_mesh3d, _ = load_ycb_obj(YCBModelsDir, data['objName'], data['objRot'], data['objTrans'])
        obj_mesh_proj = cam_intr.dot(obj_mesh3d.transpose()).transpose()
        obj_mesh2d = (obj_mesh_proj / obj_mesh_proj[:, 2:])[:, :2]        
        
        mesh3d = np.concatenate((hand_mesh3d, obj_mesh3d), axis=0)
        mesh2d = np.concatenate((hand_mesh2d, obj_mesh2d), axis=0) 
    
    return hand_object3d, hand_object2d, mesh3d, mesh2d

if __name__ == '__main__':

    mano_layer = ManoLayer(mano_root=mano_root, use_pca=False, ncomps=6, flat_hand_mean=True)
    names = ['images', 'depths', 'points2d', 'points3d', 'mesh3d', 'mesh2d']
    file_dict_train = defaultdict(list)
    file_dict_val = defaultdict(list)
    name_object_dict = {}

    val_list = ['MC6', 'SiS1', 'SiBF12', 'GPMF11', 'SM4', 'SiBF14']

    directory = f'val_size_{len(val_list)}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    count = 0
    for subject in tqdm(sorted(os.listdir(os.path.join(train)))):
        s_path = os.path.join(train, subject)
        rgb = os.path.join(s_path, 'rgb')
        depth = os.path.join(s_path, 'depth')
        meta = os.path.join(s_path, 'meta')
        
        for rgb_file in sorted(os.listdir(rgb)):
            file_number = rgb_file.split('.')[0]
            meta_file = os.path.join(meta, file_number+'.pkl')
            img_path = os.path.join(rgb, rgb_file)        
            depth_path = os.path.join(depth, file_number+'.png')        

            data = np.load(meta_file, allow_pickle=True)

            if data['handJoints3D'] is None:
                # Load previous frame's data if data is missing
                count += 1
                continue
            else:
                hand_object3d, hand_object2d, mesh3d, mesh2d = load_annotations(data, mano_layer)

      
            values = [img_path, depth_path, hand_object2d, hand_object3d, mesh3d, mesh2d]
            if subject in val_list:
                for i, name in enumerate(names):
                    file_dict_val[name].append(values[i])
            else:
                for i, name in enumerate(names):
                    file_dict_train[name].append(values[i])

    print('Total number of failures:', count)
    print("size of training dataset", len(file_dict_train['points2d']))
    print("size of validation dataset", len(file_dict_val['points2d']))

    # Appending all possible 2D points to normalize
    # points_2d_lists = [file_dict_train['hand_mesh2d'], file_dict_train['points2d'], file_dict_val['hand_mesh2d'], file_dict_val['points2d']]
    # points_2d_reshaped = []
    # for l in points_2d_lists:
    #     reshaped_arr = np.array(l).reshape((-1, 2))
    #     points_2d_reshaped.append(reshaped_arr)
    # all_points2d = np.concatenate(points_2d_reshaped)

    # # Create a scaler object to normalize 2D points
    # scaler2d = fit_scaler(all_points2d, '2d')    

    # # Appending all possible 3D points to normalize
    # points_3d_lists = [file_dict_train['hand_mesh'], file_dict_train['points3d'], file_dict_val['hand_mesh'], file_dict_val['points3d']]
    # points_3d_reshaped = []
    # for l in points_3d_lists:
    #     reshaped_arr = np.array(l).reshape((-1, 3))
    #     points_3d_reshaped.append(reshaped_arr)
    # all_points3d = np.concatenate(points_3d_reshaped)
    
    # # Create a scaler object to normalize 3D points
    # scaler3d = fit_scaler(all_points3d, '3d')


    for k, v in file_dict_train.items():
        np.save(f'{dataset_path}/{k}-train.npy', np.array(v))

    for k, v in file_dict_val.items():
        np.save(f'{dataset_path}/{k}-val.npy', np.array(v))

    file_dict_test = defaultdict(list)
    name_object_dict = {}

    # # Evaluation
    count = 0
    for subject in tqdm(os.listdir(os.path.join(evaluation))):
        s_path = os.path.join(evaluation, subject)
        rgb = os.path.join(s_path, 'rgb')
        depth = os.path.join(s_path, 'depth')
        meta = os.path.join(s_path, 'meta')
            
        for rgb_file in os.listdir(rgb):
            file_number = rgb_file.split('.')[0]
            img_path = os.path.join(rgb, rgb_file)
            depth_path = os.path.join(depth, file_number+'.png')  
            meta_file = os.path.join(meta, file_number+'.pkl')
            count+=1
            data = np.load(meta_file, allow_pickle=True)
            if data['handJoints3D'] is None:
                continue
                # hand_object3d, hand_object2d, mesh3d, mesh2d = last_hand_object3d, last_hand_object2d, last_mesh3d, last_mesh2d
            else:
                hand_object3d, hand_object2d, mesh3d, mesh2d = load_annotations(data, mano_layer, subset='test')
                # last_hand_object3d, last_hand_object2d, last_mesh3d, last_mesh2d = hand_object3d, hand_object2d, mesh3d, mesh2d
                # print(hand_object3d.shape, hand_object2d.shape, mesh3d.shape, mesh2d.shape)
      
            values = [img_path, depth_path, hand_object2d, hand_object3d, mesh3d, mesh2d]
            
            for i, name in enumerate(names):
                file_dict_test[name].append(values[i])

    for k, v in file_dict_test.items():
        np.save(f'{dataset_path}/{k}-test.npy', np.array(v))

    print("size of testing dataset", len(file_dict_test['points2d']))
    print("total testing samples:", count, "percentage:", len(file_dict_test['points2d'])/count)