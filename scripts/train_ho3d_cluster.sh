#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3 main_THOR.py \
  --root /datasets/ho3d/ \
  --output_file /checkpoints/hand-object/model- \
  --batch_size 8 \
  --gpu_number 0 \
  --photometric \
  --hid_size 128 \
  --object \
