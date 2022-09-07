import torch
import numpy as np
from dataset import Dataset
from network.GraFormer import GraFormer, adj_mx_from_edges
from common.data_utils import create_edges
from main_GraFormer import evaluate


dataset_path = '../../HOPE/datasets/ho-v3-mesh'
seq_length = 9
mesh = False
n_points = 21
obj = False
dim_model = 96 
n_layer = 5
n_head = 4
dropout = 0.25

ckpt_path = f'./checkpoint/GTNet_V3_cheb_2l-{n_points}-gt-{seq_length}/_head-4-layers-5-dim-96/_lr_step50000-lr_gamma0.9-drop_0.25/ckpt_best.pth.tar'
device = torch.device("cuda:1")

if obj:
    n_points = 29
if mesh:
    n_points = 778
    if obj:
        n_points = 1778

valset = Dataset(dataset_path, load_set='val', seq_length=seq_length, n_points=n_points, eval_pred=True)
valid_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=True, num_workers=16)  

# Create model
print("==> Creating model...")
edges = create_edges(seq_length, n_points)
adj = adj_mx_from_edges(num_pts=n_points * seq_length, edges=edges, sparse=False)
model_pos = GraFormer(adj=adj.to(device), hid_dim=dim_model, coords_dim=(2, 3), n_pts=n_points * seq_length, num_layers=n_layer, n_head=n_head, dropout=dropout).to(device)
model_pos.mask = model_pos.mask.to(device)

# Loading checkpoint
print("==> Loading checkpoint '{}'".format(ckpt_path))
ckpt = torch.load(ckpt_path)
start_epoch = ckpt['epoch']
error_best = ckpt['error']
glob_step = ckpt['step']
lr_now = ckpt['lr']
model_pos.load_state_dict(ckpt['state_dict'])
print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))


# Evaluation
print('==> Evaluating...')
errors_p1, errors_p2 = evaluate(valid_loader, model_pos, device, seq_length)
print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))