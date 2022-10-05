# HO3D
srun -K --gpus=1 \
--container-mounts=/netscratch/aboukhadra/checkpoints:/checkpoints/,/netscratch/aboukhadra/datasets:/datasets/,/ds-av/public_datasets/honotate_v3/raw:/home2/HO3D_v3,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.02-py3.sqsh \
--container-workdir="`pwd`" \
--cpus-per-gpu=8 --mem=80G --partition=A100 \
./scripts/train_ho3d_cluster.sh


# H2O
srun -K --gpus=1 \
--container-mounts=/netscratch/aboukhadra/checkpoints:/checkpoints/,/netscratch/aboukhadra/datasets:/datasets/,/ds-av/public_datasets/h2o/wds:/ds-av/public_datasets/h2o/wds,"`pwd`":"`pwd`" \
--container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.07-py3.sqsh \
--container-workdir="`pwd`" \
--cpus-per-gpu=8 --mem=120G --partition=A100 \
--task-prolog=./scripts/install.sh \
./scripts/train_h2o_cluster.sh

