# the_delorian_claim_ml

Repository to handle training and inference of the ML model for the Hackathon.

Based on the suitability of the [CarDD dataset](https://cardd-ustc.github.io/), I have opted to train a model on this dataset and, concretely, the same model they already used to solve the object detection task. Object detection consist on detecting an specific obejct in an image and mark a box in the image highlighting where the object is present. We are detecting as objects the car parts that are damaged. Those include the following 6 categories labelled in the CarDD dataset:
 - dent
 - scratch
 - crack
 - glass shatter
 - lamp broken
 - tire flat

Regarding the severity of the damage, the CarDD dataset contains the severity of the damage, but I do not know of an easy DL way to inlcude it and detect the severity as well, so it requires a further analysis (and we do not have that much time in the Hackathon).

This repo is using MMDetection, a open-source code used to train a wide variety of DL visual models.
In case you wondering, the same authors of MMDetection recommend to clone the whole repo to make use of it. Already tried to install it as a package dependency but realized they are not pushing some modules as a package that we require (mainly `configs` & the `tools` modules).

## Installation
This repo has a `.python.version` to automatically handle the corresponding python version (3.10.9) in case of having pyenv installed in your system.
In case you do not have pyenv, make sure to use Python 3.10.9.

Although I have left the `requirements.txt` file, I've followed the [installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) from the official MMDetection documentation. I leave the specific instructions I've followed:

```bash
# If you do not have virtualenv installed
pip install virtualenv
# Create a Virtual environment
virtualenv venv
# Activate the virtualenv
source venv/bin/activate
# Install main DL dependencies for MMDetection
pip install torch==2.1.0 torchvision==0.16.0
# Install a rare package manager they use called mim
pip install -U openmim
# Use mim to install mmengine & mmcv
mim install mmengine
mim install "mmcv>=2.0.0"
# Install MMDetection from its source
pip install -v -e mmdetection
```

### Download CarDD Dataset

Download the dataset zip file from the [official Drive URL](https://drive.google.com/file/d/1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx/view). 
Unzip the file and extract the `CarDD_release` folder. Then:
```bash
mkdir -p data
mv CarDD_release/CarDD_COCO data/.
```

Feel free to remove the other dataset inside the `CarDD_release` folder afterwards.

### Download the DCN pretrained model on the COCO dataset

Use the [download_pretrained_model.sh](download_pretrained_model.sh) shell script for downloading the pretrained model:
```bash
sh download_pretrained_model.sh
```

## Training

In order to train the model, we have created specific configuration files that inherit from other more complex setups to train DL models. In this case, there are 2 configuration files:
 - [DCN_plus_cfg.py](configs/car_damage/DCN_plus_cfg.py)
 - [small_DCN_plus_cfg.py](configs/car_damage/small_DCN_plus_cfg.py)

 The first one is the proper configuration to train the real [Deformable Convolution Network](https://arxiv.org/abs/1703.06211) model using the CarDD dataset on top of a pretrained DCN model already trained on the COCO dataset (a large image dataset for object detection).
To train the model you just need to execute the python [train_model.py](train_model.py) module:

```bash
python train_model.py
```
DISCLAIMER: Do not execute this, it takes ages in a regular computing node (and even much more in a laptop). This is expected to run on a machine with a GPU+CUDA available.

### Toy training
In case you wanna play with a toy example, I have created the small version of the dataset to check everything runs as expected before running the main one. Feel free to give it a try:

```bash
# Copy data folder
cp data small_data
# Remove all images from all folders not containing the pattern "*0.jpg" (Keep 1/10 fo the data)
find small_data ! -iname "*0.jpg" -delete
# Remove the metadata as well according to the same pattern
python data_reducer.py
# Execute the training script but with the small dataset
python train_small_model.py
```

## Inference
Remains to be done! Have not yet inspected in detail how to do inference of the model, which would be the core part of a service capable of serving such a service when recieving an image.

Some bullet points I think of:
 - Use the same configuration file created for the training
 - Create a `inference_model.py` module, inspired by the `train_model.py` one.
 - I inspired the `train_model.py` on the `mmdetection/tools/train.py`. I would inspect around that folder to check how to do the inference (maybe the `mmdetection/tools/test.py`??).