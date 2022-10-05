#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3.8 main_THOR.py \
  --dataset_name h2o \
  --root /ds-av/public_datasets/h2o/wds/ \
  --output_file ./checkpoints/h2o-photometric/model- \
  --batch_size 2 \
  --log_batch 10 \
  --gpu_number 1 \
  --buffer_size 10 \
  --object \
  --photometric \

  # --pretrained_model ./checkpoints/h2o-2d/model-3.pkl
