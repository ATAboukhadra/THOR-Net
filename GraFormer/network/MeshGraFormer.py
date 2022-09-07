from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
import scipy
from torch.nn.parameter import Parameter
from .ChebConv import ChebConv, _ResChebGC
from .GraFormer import GraphNet, GraAttenLayer, MultiHeadedAttention, adj_mx_from_edges, attention

def create_edges(seq_length=1, num_nodes=29):

    edges = [
        # Hand connectivity
        [0, 1], [1, 2], [2, 3], [3, 4], 
        [0, 5], [5, 6], [6, 7], [7, 8], 
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16], 
        [0, 17], [17, 18], [18, 19], [19, 20]]
    if num_nodes == 29:
        # Object connectivity
        edges.extend([
        [21, 22],[22, 24], [24, 23], [23, 21],
        [25, 26], [26, 28], [28, 27], [27, 25],
        [21, 25], [22, 26], [23, 27], [24, 28]])

    edges = torch.tensor(edges, dtype=torch.long)
    return edges

class GraphUnpool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)        

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class MeshGraFormer(nn.Module):
    def __init__(self, initial_adj, coords_dim=(2, 3), hid_dim=128, num_layers=5, n_head=4,  dropout=0.1, num_kps3d=50, num_verts=2556, adj_matrix_root='./GraFormer/adj_matrix'):
        super(MeshGraFormer, self).__init__()
        self.n_layers = num_layers
        self.initial_adj = initial_adj
        self.device = initial_adj.device
        self.num_points_levels = 3

        hid_dim_list = [hid_dim, hid_dim // 4, hid_dim // 16, coords_dim[1]]

        level1 = round(num_verts / 16)    
        
        if num_kps3d == 50: #i.e. h2o 2 hands and object
            level2 = num_verts // 4 - 1
        else:
            level2 = num_verts // 4
            
        obj = ''
        if num_kps3d == 50 or num_kps3d == 29:
            obj = 'Object'

        points_levels = [num_kps3d, level1, level2, num_verts]
        self.mask = [torch.tensor([[[True] * points_levels[i]]]).to(self.device) for i in range(3)]
        
        self.adj = [initial_adj.to(self.device)]
        self.adj.extend([torch.from_numpy(scipy.sparse.load_npz(f'{adj_matrix_root}/hand{obj}{points_levels[i]}.npz').toarray()).float().to(self.device) for i in range(1, 4)])
                
        gconv_inputs = []
        gconv_layers = []
        attention_layers = []
        unpooling_layers = []
        gconv_outputs = []
        c = copy.deepcopy
        
        self.gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim_list[0], K=2)

        for i in range(self.num_points_levels):
            hid_dim = hid_dim_list[i]
            
            gconv_inputs.append(ChebConv(in_c=hid_dim, out_c=hid_dim, K=2))
            
            attn = MultiHeadedAttention(n_head, hid_dim)
            gcn = GraphNet(in_features=hid_dim, out_features=hid_dim, n_pts=points_levels[i])    
            attention_layer = []
            gconv_layer = []

            for j in range(num_layers):
                gconv_layer = _ResChebGC(adj=self.adj[i], input_dim=hid_dim, output_dim=hid_dim, hid_dim=hid_dim, p_dropout=0.1)
                attention_layer = GraAttenLayer(hid_dim, c(attn), c(gcn), dropout)
                attention_layers.append(attention_layer)
                gconv_layers.append(gconv_layer)
            
            gconv_outputs.append(ChebConv(in_c=hid_dim, out_c=hid_dim_list[i+1], K=2))
            unpooling_layers.append(GraphUnpool(points_levels[i], points_levels[i+1]))
            
        self.gconv_inputs = nn.ModuleList(gconv_inputs)
        self.gconv_layers = nn.ModuleList(gconv_layers)
        self.atten_layers = nn.ModuleList(attention_layers)
        self.gconv_output = nn.ModuleList(gconv_outputs)

        self.unpooling_layer = nn.ModuleList(unpooling_layers)


    def forward(self, x):
        out = self.gconv_input(x, self.adj[0])
        for i in range(self.num_points_levels):
            out = self.gconv_inputs[i](out, self.adj[i])
            for j in range(self.n_layers):
                out = self.atten_layers[i * self.n_layers + j](out, self.mask[i])
                out = self.gconv_layers[i * self.n_layers + j](out)
            out = self.gconv_output[i](out, self.adj[i])
            out = self.unpooling_layer[i](out)
        
        return out

if __name__ == '__main__':
    features = 3 + 1024
    num_points = 29
    x = torch.zeros((1, num_points, features))
    edges = create_edges(1, num_points)
    initial_adj = adj_mx_from_edges(num_pts=num_points, edges=edges, sparse=False)
    mesh_graformer = MeshGraFormer(initial_adj, coords_dim=(features,3), n_pts=1778)
    output = mesh_graformer(x)
    print(output.shape)

