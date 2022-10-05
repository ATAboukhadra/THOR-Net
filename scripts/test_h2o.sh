export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3.8 test_THOR.py \
 --dataset_name h2o \
 --root /ds-av/public_datasets/h2o/wds/ \
 --checkpoint_folder h2o-photometric \
 --checkpoint_id 13 \
 --split val \
 --seq rgb \
 --gpu_number 1 \
 --batch_size 1 \
 --object \
 --photometric \
 --buffer_size 10 \
#  --visualize \