export CUDA_DEVICE_ORDER=PCI_BUS_ID

python3.8 test_THOR.py \
 --dataset_name ho3d \
 --root ./datasets/ho3d/ \
 --checkpoint_folder ho3d-photometric \
 --checkpoint_id 18 \
<<<<<<< HEAD
 --split test \
 --seq SM1 \
=======
 --split val \
 --seq rgb \
>>>>>>> 6e0d4283c835934e45a2f1e5613843d73cfb8c21
 --gpu_number 1 \
 --batch_size 1 \
 --object \
 --visualize \
 --hid_size 96 \
 --photometric \

