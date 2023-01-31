#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3.8 main_THOR.py \
  --root ./datasets/ho3d/ \
  --output_file ./checkpoints/hand-object/model- \
  --batch_size 2 \
  --log_batch 10 \
  --gpu_number 1 \
  --hid_size 96 \
  # --object \
