from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dataset import Dataset
from torch.utils.data import DataLoader
from progress.bar import Bar
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.data_utils import fetch, read_3d_data, create_2d_data, create_edges
from common.generators import PoseGenerator
from common.loss import mpjpe, p_mpjpe
from network.GraFormer import GraFormer, adj_mx_from_edges
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset_path', default='../HOPE/datasets/ho-v3-mesh/', type=str, metavar='NAME', help='target dataset')
    # sh_ft_h36m, gt
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')
    parser.add_argument('--gpu_number', default=1, type=int, help='The id of the GPU')

    parser.add_argument('--n_head', type=int, default=4, help='num head')
    parser.add_argument('--dim_model', type=int, default=96, help='dim model')
    parser.add_argument('--n_layer', type=int, default=5, help='num layer')
    parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')
    
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=1, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=50000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='gamma of learning rate decay')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    parser.add_argument('--seq_length', default=1, type=int, metavar='FACTOR', help='how many adjacent poses to include?')
    parser.add_argument("--mesh", action='store_true', help="Generate mesh for hand and object from 2D mesh")
    parser.add_argument("--obj", action='store_true', help="Generate pose of shape for object too")

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args

def main(args):
    global src_mask
    n_points = 21
    if args.obj:
        n_points = 29
    if args.mesh:
        n_points = 778
        if args.obj:
            n_points = 1778

    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('')

    cudnn.benchmark = True
    device = torch.device(f"cuda:{args.gpu_number}")

    # Create model
    print("==> Creating model...")

    edges = create_edges(args.seq_length, n_points)
    adj = adj_mx_from_edges(num_pts=n_points * args.seq_length, edges=edges, sparse=False)
    model_pos = GraFormer(adj=adj.to(device), hid_dim=args.dim_model, coords_dim=(2, 3), n_pts=n_points * args.seq_length, num_layers=args.n_layer, n_head=args.n_head, dropout=args.dropout).to(device)
    model_pos.mask = model_pos.mask.to(device)

    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        # ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())
        ckpt_dir_path = path.join(args.checkpoint, 'GTNet_V3_cheb_2l-' + str(n_points) + '-' + args.keypoints + '-' + str(args.seq_length),
                                  '_head-%s' % args.n_head + '-layers-%s' % args.n_layer + '-dim-%s' % args.dim_model,
                                  '_lr_step%s' % args.lr_decay + '-lr_gamma%s' % args.lr_gamma + '-drop_%s' % args.dropout)
        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))
        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])


    if args.evaluate:
        print('==> Evaluating...')

        valset = Dataset(args.dataset_path, load_set='val', seq_length=args.seq_length, n_points=n_points)
        valid_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=16)    
        errors_p1, errors_p2 = evaluate(valid_loader, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        exit(0)

    trainset = Dataset(args.dataset_path, load_set='train', seq_length=args.seq_length, n_points=n_points)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=16)    
    
    valset = Dataset(args.dataset_path, load_set='val', seq_length=args.seq_length, n_points=n_points)
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True, num_workers=16)    
    
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now, glob_step, args.lr_decay, args.lr_gamma)
        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)
        
        
    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return


def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(data_loader))
    # dl_ = tqdm(data_loader)
    for i, (inputs_2d, targets_3d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device).float(), inputs_2d.to(device).float()
        # inputs_2d_rcnn = inputs_2d_rcnn.to(device).float()
        outputs_3d = model_pos(inputs_2d) # ---------------

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        # loss_3d_pos = criterion(outputs_3d, inputs_2d)

        loss_3d_pos.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_loss_3d_pos.update(loss_3d_pos.item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_loss_3d_pos.avg)
        bar.next()
        
    bar.finish()
    return epoch_loss_3d_pos.avg, lr_now, step


def evaluate(data_loader, model_pos, device, seq_length=1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (inputs_2d, targets_3d) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.float().to(device)
        # targets_3d = targets_3d.float().to(device)
        
        # inputs_2d_rcnn = inputs_2d_rcnn.float().to(device)

        # [:, -29:] i.e. Use last pose if more than 1 pose, otherwise use the only pose
        offset=0
        if seq_length > 1:
            offset = seq_length - 1
        outputs_3d = model_pos(inputs_2d)[:, 21*offset:21*(offset+1)]
        targets_3d = targets_3d[:, 21*offset:21*(offset+1)]
        # outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)
        # print(targets_3d[:, -29:].shape)
        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d.float().to(device)).item(), num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d, targets_3d).item(), num_poses)

        
        # inputs_2d = inputs_2d.cpu().detach().numpy()

        # epoch_loss_3d_pos.update(mpjpe(outputs_3d, inputs_2d[:, -29:]).item(), num_poses)        
        # epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d, inputs_2d[:, -29:]).item(), num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())
