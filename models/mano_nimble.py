import numpy as np 
import torch
from torch import nn
from models.NIMBLE_model.NIMBLELayer import NIMBLELayer
from models.NIMBLE_model.utils import batch_to_tensor_device
from models.faster_rcnn import TwoMLPHead

assets_path = 'models/NIMBLE_model/assets'


class MeshEncoder(nn.Module):
    def __init__(self):
        super(MeshEncoder, self).__init__()
        self.encoder_layer = TwoMLPHead(778 * 3, 50)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        out = self.encoder_layer(x)
        pose, shape = out[:, :30], out[:, 30:]
        return pose, shape

class ManoNimbleAE(nn.Module):
    def __init__(self, device):
        super(ManoNimbleAE, self).__init__()

        # device = 'cuda'
        self.encoder = MeshEncoder()

        pm_dict_name = f"{assets_path}/NIMBLE_DICT_9137.pkl"
        tex_dict_name = f"{assets_path}/NIMBLE_TEX_DICT.pkl"

        pm_dict = np.load(pm_dict_name, allow_pickle=True)
        pm_dict = batch_to_tensor_device(pm_dict, device)

        tex_dict = np.load(tex_dict_name, allow_pickle=True)
        tex_dict = batch_to_tensor_device(tex_dict, device)

        # nimble_mano_vreg = np.load("assets/NIMBLE_MANO_VREG.pkl", allow_pickle=True)
        # nimble_mano_vreg = batch_to_tensor_device(nimble_mano_vreg, device)
        self.nimble_layer = NIMBLELayer(pm_dict, tex_dict, device, use_pose_pca=True, pose_ncomp=30, shape_ncomp=20)
        

    def forward(self, x):
        bn = x.shape[0]
        hand_mesh = x[:, :778, :3]
        pose_param, shape_param = self.encoder(hand_mesh)
        tex_param = torch.rand(bn, 10).to(hand_mesh.device) - 0.5

        skin_v, _, _, _, tex_img = self.nimble_layer(pose_param, shape_param, tex_param, handle_collision=True)


        return skin_v, tex_img