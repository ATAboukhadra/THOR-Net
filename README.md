# THOR-Net

This repo contains the PyTorch implementation for Two-Hands-Object reconstruction network (THOR-Net).

## Dependencies

```
manopth==0.0.1
matplotlib==3.3.4
numpy==1.13.3
opencv_python==4.5.3.56
Pillow==9.2.0
pymeshlab==2021.10
scikit_image==0.17.2
scipy==0.19.1
skimage==0.0
torch==1.10.1
torchvision==0.11.2
tqdm==4.62.3
```

## Step 1: Create dataset files
datasets/make_data.py creates the dataset files for HO3D by creating train-val-test splits and preprocess them to problem needs. 

Adapt the variables (root and mano_root) to the HO3D dataset path (downloadable through [HO3D](https://www.tugraz.at/index.php?id=40231)) and mano models path (downloadable through [MANO](https://mano.is.tue.mpg.de/)) 

running the following command will create the dataset files and store them in ./datasets/ho3d/

```
mkdir datasets/ho3d/
python3 datasets/make_data.py --root /path/to/HO3D --mano_root /path/to/MANO --dataset_path ./datasets/ho3d/

```
## Step 2: Train default model

The script (scripts/run_train_rcnn.sh) trains a model from scratch and has the path to the prepared dataset files and the path where the checkpoints should be stored. 

For more detailed description of different parameters check utils/options.py

```
mkdir checkpoints/hand-object
chmod a+x ./scripts/train_ho3d.sh
./scripts/train_ho3d.sh
```

## Step 3: Evaluate and Visualize model
The script (scripts/test_ho3d.sh) produces visualizations for the outputs of a pretrained specified model on a --split which could be train or val or test.

The pretrained model is located at --checkpoint_folder and has --checkpoint_id which corresponds to the epoch number.


To disable the visualization and run inference only (with evaluation if GT exists in case of a train or val split) remove --visualization flag from the script.

```
./scripts/visualize_rcnn.sh
```
