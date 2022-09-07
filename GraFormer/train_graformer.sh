#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_GraFormer.py \
    --dataset_path ../../HOPE/datasets/ho-v3-mesh \
    --gpu_number 1 \
    --seq_length 1 \
    --evaluate ./checkpoint/GTNet_V3_cheb_2l-21-gt-1/_head-4-layers-5-dim-96/_lr_step50000-lr_gamma0.9-drop_0.25/ckpt_best.pth.tar \
    # --resume ./checkpoint/GTNet_V3_cheb_2l-21-gt-1/_head-4-layers-5-dim-96/_lr_step50000-lr_gamma0.9-drop_0.25/ckpt_best.pth.tar
    # --obj \
    # --mesh 