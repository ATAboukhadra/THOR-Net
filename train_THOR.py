# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

""" import libraries"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import os

from utils.options import parse_args_function
from utils.utils import freeze_component, calculate_keypoints, create_loader
from utils.h2o_utils.h2o_dataset_utils import load_tar_split
from utils.h2o_utils.h2o_preprocessing_utils import MyPreprocessor

from models.keypoint_rcnn import keypointrcnn_resnet50_fpn

torch.multiprocessing.set_sharing_strategy('file_system')

args = parse_args_function()

# Define device
device = torch.device(f'cuda:{args.gpu_number[0]}' if torch.cuda.is_available() else 'cpu')

use_cuda = False
if torch.cuda.is_available():
    use_cuda = True

num_kps2d, num_kps3d, num_verts = calculate_keypoints(args.dataset_name, args.object)

""" Configure a log """

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(args.output_file[:-6], 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


""" load datasets """

if args.dataset_name.lower() == 'h2o':
    annotation_components = ['cam_pose', 'hand_pose', 'hand_pose_mano', 'obj_pose', 'obj_pose_rt', 'action_label', 'verb_label']
    my_preprocessor = MyPreprocessor('../mano_v1_2/models/', 'datasets/objects/mesh_1000/', args.root)
    h2o_data_dir = os.path.join(args.root, 'shards')
    train_input_tar_lists, train_annotation_tar_files = load_tar_split(h2o_data_dir, 'train')    
    val_input_tar_lists, val_annotation_tar_files = load_tar_split(h2o_data_dir, 'val')   
    num_classes = 4
    graph_input = 'coords'
else: # i.e. HO3D
    trainloader = create_loader(args.dataset_name, args.root, 'train', batch_size=args.batch_size, num_kps3d=num_kps3d, num_verts=num_verts)
    valloader = create_loader(args.dataset_name, args.root, 'val', batch_size=args.batch_size)
    num_classes = 2 
    graph_input = 'heatmaps'

""" load model """
model = keypointrcnn_resnet50_fpn(num_kps2d=num_kps2d, num_kps3d=num_kps3d, num_verts=num_verts, num_classes=num_classes, 
                                rpn_post_nms_top_n_train=1, rpn_post_nms_top_n_test=1, 
                                device=device, num_features=args.num_features, 
                                photometric=args.photometric, graph_input=graph_input)
print('Keypoint RCNN is loaded')

if torch.cuda.is_available():
    model = model.cuda(args.gpu_number[0])
    model = nn.DataParallel(model, device_ids=args.gpu_number)

""" load saved model"""

if args.pretrained_model != '':
    model.load_state_dict(torch.load(args.pretrained_model, map_location=f'cuda:{args.gpu_number[0]}'))
    losses = np.load(args.pretrained_model[:-4] + '-losses.npy').tolist()
    start = len(losses)
else:
    losses = []
    start = 0

"""define optimizer"""

criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)
scheduler.last_epoch = start

keys = ['boxes', 'labels', 'keypoints', 'keypoints3d', 'mesh3d', 'palm']

""" training """

logging.info('Begin training the network...')

for epoch in range(start, args.num_iterations):  # loop over the dataset multiple times
    
    train_loss2d = 0.0
    running_loss2d = 0.0
    running_loss3d = 0.0
    running_mesh_loss3d = 0.0
    running_photometric_loss = 0.0
    
    if 'h2o' in args.dataset_name.lower():
        h2o_info = (train_input_tar_lists, train_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
        trainloader = create_loader(args.dataset_name, h2o_data_dir, 'train', args.batch_size, h2o_info=h2o_info)

    for i, tr_data in enumerate(trainloader):
        
        # get the inputs
        data_dict = tr_data
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
        inputs = [t['inputs'].to(device) for t in data_dict]
        loss_dict = model(inputs, targets)
        
        # Calculate Loss
        loss = sum(loss for _, loss in loss_dict.items())
        
        # Backpropagate
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss2d += loss_dict['loss_keypoint'].data
        running_loss2d += loss_dict['loss_keypoint'].data
        running_loss3d += loss_dict['loss_keypoint3d'].data
        running_mesh_loss3d += loss_dict['loss_mesh3d'].data
        if 'loss_photometric' in loss_dict.keys():
            running_photometric_loss += loss_dict['loss_photometric'].data

        if (i+1) % args.log_batch == 0:    # print every log_iter mini-batches
            logging.info('[%d, %5d] loss 2d: %.4f, loss 3d: %.4f, mesh loss 3d:%.4f, photometric loss: %.4f' % 
            (epoch + 1, i + 1, running_loss2d / args.log_batch, running_loss3d / args.log_batch, 
            running_mesh_loss3d / args.log_batch, running_photometric_loss / args.log_batch))
            running_mesh_loss3d = 0.0
            running_loss2d = 0.0
            running_loss3d = 0.0
            running_photometric_loss = 0.0

    losses.append((train_loss2d / (i+1)).cpu().numpy())
    
    if (epoch+1) % args.snapshot_epoch == 0:
        torch.save(model.state_dict(), args.output_file+str(epoch+1)+'.pkl')
        np.save(args.output_file+str(epoch+1)+'-losses.npy', np.array(losses))

    if (epoch+1) % args.val_epoch == 0:
        val_loss2d = 0.0
        val_loss3d = 0.0
        val_mesh_loss3d = 0.0
        val_photometric_loss = 0.0
        
        # model.module.transform.training = False
        
        if 'h2o' in args.dataset_name.lower():
            h2o_info = (val_input_tar_lists, val_annotation_tar_files, annotation_components, args.buffer_size, my_preprocessor)
            valloader = create_loader(args.dataset_name, h2o_data_dir, 'val', args.batch_size, h2o_info)

        for v, val_data in enumerate(valloader):
            
            # get the inputs
            data_dict = val_data
        
            # wrap them in Variable
            targets = [{k: v.to(device) for k, v in t.items() if k in keys} for t in data_dict]
            inputs = [t['inputs'].to(device) for t in data_dict]    
            loss_dict = model(inputs, targets)
            
            val_loss2d += loss_dict['loss_keypoint'].data
            val_loss3d += loss_dict['loss_keypoint3d'].data
            val_mesh_loss3d += loss_dict['loss_mesh3d'].data
            if 'loss_photometric' in loss_dict.keys():
                running_photometric_loss += loss_dict['loss_photometric'].data
        
        # model.module.transform.training = True
        
        logging.info('val loss 2d: %.4f, val loss 3d: %.4f, val mesh loss 3d: %.4f, val photometric loss: %.4f' % 
                    (val_loss2d / (v+1), val_loss3d / (v+1), val_mesh_loss3d / (v+1), val_photometric_loss / (v+1)))        
    
    if args.freeze and epoch == 0:
        logging.info('Freezing Keypoint RCNN ..')            
        freeze_component(model.module.backbone)
        freeze_component(model.module.rpn)
        freeze_component(model.module.roi_heads)

    # Decay Learning Rate
    scheduler.step()

logging.info('Finished Training')