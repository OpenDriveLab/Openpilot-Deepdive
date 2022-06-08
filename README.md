# <p align="center">Openpilot-Deepdive</p>
**<p align="center">Level 2 Autonomous Driving on a Single Device: Diving into the Devils of Openpilot</p>**
![image](https://github.com/OpenPerceptionX/Openpilot-Deepdive/blob/main/imgs/arXiv_fig1.png)
[**Webpage**](https://sites.google.com/view/openpilot-deepdive/home) | [**Paper**]() | [**Zhihu**](https://www.zhihu.com/people/PerceptionX)
***
# Introduction

This repository is the PyTorch implementation for our Openpilot-Deepdive.
In contrast to most traditional autonomous driving solutions where the perception, prediction, and planning module are apart, [Openpilot](https://github.com/commaai/openpilot) uses an end-to-end neural network to predict the trajectory directly from the camera images, which is called Supercombo. We try to reimplement the training details and test the pipeline on public benchmarks. Experimental results of OP-Deepdive on nuScenes, Comma2k19, CARLA, and in-house realistic scenarios (collected in Shanghai) verify that a low-cost device can indeed achieve most L2 functionalities and be on par with the original Supercombo model.

* [Directory Structure](#directory-structure) 
* [Changelog](#changelog) 
* [Quick Start](#quick-start-examples) 
    * [Installation](#installation)
    * [Dataset](#dataset)
    * [Training](#training)
    * [Demo](#demo)
* [Baselines](#baselines)
* [Citation](#citation) 
* [License](#license)  

***
# Directory Structure

```
Openpilot-Deepdive 
├── tools           - Generate image data from Comma2k19 and nuScenes.  
├── utils_comma2k19 - Comma2k19 dataset processing code.  
├── data
      ├── nuscenes  -> link to the nusSenes-all dataset
      ├── comma2k19 -> link to the Comma2k19 dataset
```
***
# Changelog

2022-6-10: We released the v1.0 code for Openpilot-reimplementation.

***
# Quick Start Examples
Before starting, we refer you to read the [arXiv]() to understand the details of our work.
## Installation
Clone repo and install requirements.txt in a [Python>=3.7.0](https://www.python.org/) environment, including [PyTorch>=1.7](https://pytorch.org/get-started/locally/).

```
git clone https://github.com/OpenPerceptionX/Openpilot-Deepdive.git  # clone
cd Openpilot-Deepdive
pip install -r requirements.txt  # install
```
## Dataset
We train and evaluate our model on two datasets, [nuScenes](https://www.nuscenes.org/nuscenes) and [Comma2k19](https://github.com/commaai/comma2k19).
The table shows some key features of them.

| Dataset     | Raw<br>FPS (Hz)  | Aligned&<br>FPS (Hz) | Length Per<br>Sequence<br>(Frames/Second) | Altogether<br>Length<br>(Minutes) | Scenario | Locations
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|
| nuScenes | 12 | 2 | 40 / 20 | 330 | Street | America<br>Singapore |  
| Comma2k19  | 20 | 20 | 1000 / 60 | 2000 | Highway | America | 

## Training
By default, the batch size and the learning rate are set to be 48 and 1e-4, respectively. A gradient clip of value 1.0 is applied. During training, you can use 4 or 8 NVIDIA V100 GPUs (Multi-GPU times faster). Since there is a GRU module, you need to initialize its hidden state by filling zeros. When using 8 V100 GPUs, it takes approximate 120 hours to train 100 epochs on Comma2k19 dataset . On a single NVIDIA GTX 1080 GPU, the network can inference at a speed of 100 FPS.  
Configure the above default parameters，and run the following code:
```
# using slurm to init
srun -p $PARTITION$ --job-name=openpilot --mpi=pmi2 -n $NUM_GPUS$ --gres=gpu:$NUM_GPUS$ --ntasks-per-node=$NUM_GPUS$ python main.py --batch_size=$BATCH_SIZE$ --nepochs=$NUM_EPOCHS$

```

## Demo
See more demo and test cases on our [webpage](https://sites.google.com/view/openpilot-deepdive/home).

<img src="https://github.com/OpenPerceptionX/Openpilot-Deepdive/blob/main/imgs/demo01.png" width="600px">

***
# Baselines
Here we list several baselines to perform trajectory prediction task. You are welcome to pull request and add your work here!

TODO

***
# Citation
Please use the following citation when referencing our repo or [arXiv]().
```
@article{chen2022op,
   title={Level 2 Autonomous Driving on a Single Device: Diving into the Devils of Openpilot},
   author={Chen, Li and Tang, Tutian and Cai, Zhitian and Li, Yang and Wu Penghao and Li, Hongyang and Shi, Jianping and Qiao, Yu and Yan, Junchi},
   journal={arXiv preprint arXiv:},
   year={2022}
}
```
***
# License
All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
***

