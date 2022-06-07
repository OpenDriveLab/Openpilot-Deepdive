# <p align="center">Supercombo-reimplementation</p>
**<p align="center">Level 2 Autonomous Driving on a Single Device: Diving into the Devils of Openpilot</p>**
![image](https://github.com/OpenPerceptionX/openpilot-reimplementation/blob/main/imgs/arXiv_fig1.png)
> arXiv link: [arXiv](https://www.overleaf.com/project/626954666328026cc3d14c98)
***
# Introduction

This repository is the PyTorch implementation for our Openpilot-reimplementation.
In contrast to most traditional autonomous driving solutions where the perception, prediction, and planning module are apart, [openpilot](https://github.com/commaai/openpilot) uses an end-to-end neural network to predict the trajectory directly from the camera images, which is called supercombo. we reimplement the essential supercombo model from scratch, and test it on public datasets. Experiments prove that both the original openpilot and our reimplemented model can perform well on highway scenarios.

* [Directory Structure](#directory-structure) 
* [Changelog](#changelog) 
* [Quick Start](#quick-start-examples) 
    * [Installation](#installation)
    * [Dataset](#dataset)
    * [Training](#training)
    * [Demo](#demo)
* [Citation](#citation) 
* [License](#license)  

***
# Directory Structure

```
openpilot-reimplementation 
├── tools           - Generate image data from comma2k19 and nuScenes.  
├── utils_comma2k19 - comma2k19 dataset processing code.  
├── data            - link to the nuscenes-all/comma2k19 dataset 
```
***
# Changelog

2022-6-10: We released the v1.0 code for Openpilot-reimplementation.

***
# Quick Start Examples
Before starting, you can read the [arXiv](https://www.overleaf.com/project/626954666328026cc3d14c98) paper to understand the details of our work.
## Installation
Clone repo and install requirements.txt in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/).

```
https://github.com/OpenPerceptionX/openpilot-reimplementation.git  # clone
cd openpilot-reimplementation
pip install -r requirements.txt  # install

```
## Dataset
We train and evaluate our model on two datasets, [nuscenes](https://www.nuscenes.org/nuscenes) and [Comma2k19](https://github.com/commaai/comma2k19).
The table shows some key features of them.
![dataset_cmp](https://github.com/OpenPerceptionX/openpilot-reimplementation/blob/main/imgs/dataset_cmp.png)

## Training
By default, the batch size and the learning rate are set to be 48 and 1e-4, respectively. A gradient clip of value 1.0 is applied. During training, you can use 4 or 8 NVIDIA V100 GPUs(Multi-GPU times faster). Since there is a GRU module, you need to initialize its hidden state by filling zeros. When using 8 V100 GPUs, it takes approximate 120 hours to train 100 epochs on Comma2k19 dataset . On a single NVIDIA GTX 1080 GPU, the network can inference at a speed of 100 FPS.  
Configure the above default parameters，and run the following code:
```
# using slurm to init
srun -p $PARTITION$ --job-name=openpilot --mpi=pmi2 -n $NUM_GPUS$ --gres=gpu:$NUM_GPUS$ --ntasks-per-node=$NUM_GPUS$ python main.py --batch_size=$BATCH_SIZE$ --nepochs=$NUM_EPOCHS$

```

## Demo
<img src="https://github.com/OpenPerceptionX/openpilot-reimplementation/blob/main/imgs/demo01.png" width="600px">

***
# Citation
Please use the following citation when referencing our repo or [arXiv](https://www.overleaf.com/project/626954666328026cc3d14c98).
```
TODO

```
***
# License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
***

