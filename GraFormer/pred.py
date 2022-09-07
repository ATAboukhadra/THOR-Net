from __future__ import print_function, unicode_literals

import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import json
import pickle
import re
from tqdm import tqdm
from torch.autograd import Variable
from common.data_utils import create_edges
from network.GraFormer import GraFormer, adj_mx_from_edges

def load_obj_pose(data, subset='train'):
    coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    
    cam_intr = data['camMat']
    obj_pose = data['objCorners3D']
    
    # Convert to non-OpenGL coordinates
    obj_pose = obj_pose.dot(coordChangeMat.T)
    
    # Project from 3D world to Camera coordinates using the camera matrix  
    obj_pose_proj = cam_intr.dot(obj_pose.transpose()).transpose()
    obj_pose2d = (obj_pose_proj / obj_pose_proj[:, 2:])[:, :2]
    return obj_pose2d
    
def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'train':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'

def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def create_sequence(rcnn_dict, path, seq_length=1, n_points=21):
        frame_num = int(path.split('/')[-1].split('.')[0])
        point2d_seq = np.zeros((seq_length, n_points, 2))
        
        missing_frames = False
        
        for i in range(0, seq_length):
            if frame_num - i < 0:
                missing_frames = True
                break
            new_frame_num = '%04d' % (frame_num-i)
            new_path = re.sub('\d{4}', new_frame_num, path)    
            if new_path in rcnn_dict.keys():
                point2d_seq[-i-1] = rcnn_dict[new_path][:n_points, :2] 
                last_pose = -i-1
            else: # Replicate the last pose in case of missing information
                point2d_seq[-i-1] = point2d_seq[last_pose]

        if missing_frames:
            n_missing_frames = seq_length - i
            point2d_seq[0:-i] = np.tile(point2d_seq[-i], (n_missing_frames, 1, 1)) 
        
        point2d_seq = point2d_seq.reshape((seq_length * n_points, 2))

        return point2d_seq

def main(base_path, pred_out_path, pred_func, version, model, set_name=None, mesh=False, seq_length=1):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'
    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()

    # read list of evaluation files
    with open(os.path.join(base_path, set_name+'.txt')) as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]

    predictions_dict = pickle.load(open('./rcnn_outputs/rcnn_outputs_21_test.pkl', 'rb'))
    
    # iterate over the dataset once
    xyz = []
    verts = []
    for idx in tqdm(range(db_size(set_name, version))):
        if idx >= db_size(set_name, version):
            break

        seq_name = file_list[idx].split('/')[0]
        file_id = file_list[idx].split('/')[1]
        
        rgb_path = os.path.join(base_path, set_name, seq_name, 'rgb', file_id + '.jpg')
        meta_path = os.path.join(base_path, set_name, seq_name, 'meta', file_id + '.pkl')
        
        aux_info = load_pickle_data(meta_path)
        # obj_point2d = load_obj_pose(aux_info, subset='test')
    
        if mesh:
            # inputs2d = np.zeros((778, 2))
            inputs2d = predictions_dict[rgb_path][:, 2]
        else:
            # inputs2d = np.zeros((21, 2))
            if seq_length == 1:
                inputs2d = predictions_dict[rgb_path][:21, :2]
            else:
                inputs2d = create_sequence(predictions_dict, rgb_path, seq_length=seq_length)
            # inputs2d[21:] = obj_point2d
            inputs2d = torch.from_numpy(inputs2d)
        
        # use some algorithm for prediction
        xyz, verts = pred_func(model, inputs2d)
        
        # simple check if xyz and verts are in opengl coordinate system
        if np.all(xyz[:,2] > 0) or np.all(verts[:,2]>0):
            print(seq_name, file_id, xyz)
            raise Exception('It appears the pose estimates are not in OpenGL coordinate system. Please read README.txt in dataset folder. Aborting!')


        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump(
            [
                xyz_pred_list,
                verts_pred_list
            ], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))


def pred_template(model, inputs_2d):
    """ Predict joints and vertices from a given sample.
        img: (640, 480, 3) RGB image.
        aux_info: dictionary containing hand bounding box, camera matrix and root joint 3D location
    """

    # src_mask = torch.tensor([[[True] * 21]]).cuda()

    inputs_2d = Variable(inputs_2d.unsqueeze(axis=0)).float()
    if torch.cuda.is_available():
        inputs_2d = inputs_2d.cuda(device=1)

    outputs3d = model(inputs_2d) # ---------------
    xyz = outputs3d.cpu().detach().numpy()[0][-21:]
    verts = np.zeros((778, 3))

    # OpenGL coordinates and reordering
    order_idx = np.argsort(np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]))
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
    
    xyz = xyz.dot(coord_change_mat.T)[order_idx] / 1000
    verts = verts.dot(coord_change_mat.T) / 1000

    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('--base_path', type=str, default='/home2/HO3D_v3', help='Path to where the HO3D dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json', help='File to save the predictions.')
    parser.add_argument('--version', type=str, choices=['v2', 'v3'], help='version number')
    parser.add_argument('--checkpoint_folder', default='GTNet_V3_cheb_2l-21-gt-9/_head-4-layers-5-dim-96/_lr_step50000-lr_gamma0.9-drop_0.25', help='the folder of the pretrained model')
    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--seq_length', type=int, default=1, help='Sequence length')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    args = parser.parse_args()
    
    # create edges
    edges = create_edges(num_nodes=21, seq_length=args.seq_length)
    adj = adj_mx_from_edges(num_pts=21 * args.seq_length, edges=edges, sparse=False)
    
    # define model
    model = GraFormer(adj=adj.cuda(1), hid_dim=args.dim_model, coords_dim=(2, 3), n_pts=21 * args.seq_length, num_layers=args.n_layer, n_head=args.n_head, dropout=args.dropout).cuda()

    # Load pretrained model
    pretrained_model = f'./checkpoint/{args.checkpoint_folder}/ckpt_best.pth.tar'
    print("==> Loading checkpoint '{}'".format(pretrained_model))
    ckpt = torch.load(pretrained_model)
    model.load_state_dict(ckpt['state_dict'])
        
    if torch.cuda.is_available():
        model = model.cuda(1)
        model.mask = model.mask.cuda(1)
        model = nn.DataParallel(model, device_ids=[1])
    
    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model.eval()
    
    # call with a predictor function
    main(
        args.base_path,
        args.out,
        pred_func=pred_template,
        set_name='evaluation',
        version=args.version,
        model=model,
        seq_length=args.seq_length
    )

