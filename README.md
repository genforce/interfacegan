# InterfaceGAN++: Exploring the limits of InterfaceGAN

> Authors: [Apavou Clément](https://github.com/clementapa) & [Belkada Younes](https://github.com/younesbelkada)

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![pytorch 1.10.2](https://img.shields.io/badge/pytorch-1.10.2-green.svg?style=plastic)
![sklearn 0.21.2](https://img.shields.io/badge/sklearn-0.21.2-green.svg?style=plastic)

<p float="left">
  <img src="images/bald2.gif" width="200" />
  <img src="images/blond.gif" width="200" /> 
  <img src="images/makeup.gif" width="200" /> 
  <img src="images/gray_hair.gif" width="200" /> 
</p>

> From left to right - Images generated using styleGAN and the boundaries *Bald*, *Blond*, *Heavy_Makeup*, *Gray_Hair*

This the the repository to a project related to the [*Introduction to Numerical Imaging*](https://delon.wp.imt.fr/enseignement/mva-introduction-a-limagerie-numerique/) (*i.e, Introduction à l'Imagerie Numérique* in French), given by the MVA Masters program at ENS-Paris Saclay. The project and repository is based on the work from [Shen et al.](https://github.com/younesbelkada/interfacegan/blob/master/README_old.md), and fully supports their codebase. You can refer to the [original README](https://github.com/younesbelkada/interfacegan/blob/master/README_old.md)) to reproduce their results.

- [Introduction](#introduction)
- [:fire: Additional features](#-fire--additional-features)
- [:hammer: Training an attribute detection classifier](#-hammer--training-an-attribute-detection-classifier)
- [:star: Generate images using StyleGAN & StyleGAN2 & StyleGAN3](#-star--generate-images-using-stylegan---stylegan2---stylegan3)
  * [:movie_camera: Get the pretrained StyleGAN](#-movie-camera--get-the-pretrained-stylegan)
  * [:movie_camera: Get the pretrained StyleGAN2](#-movie-camera--get-the-pretrained-stylegan2)
  * [:movie_camera: Get the pretrained StyleGAN3](#-movie-camera--get-the-pretrained-stylegan3)
  * [:art: Run the generation script](#-art--run-the-generation-script)
- [:pencil2: Edit generated images](#-pencil2--edit-generated-images)
  * [Examples](#examples)
    + [StyleGAN](#stylegan)
    + [StyleGAN2](#stylegan2)
    + [StyleGAN3](#stylegan3)

## Introduction

> In this repository, we propose an approach, termed as InterFaceGAN++, for semantic face editing based on the work from Shen et al. Specifically, we leverage the ideas from the previous work, by applying the method for new face attributes, and also for StyleGAN3. We qualitatively explain that moving the latent vector toward the trained boundaries leads in many cases to keeping the semantic information of the generated images (by preserving its local structure) and modify the desired attribute, thus helps to demonstrate the disentangled property of the styleGANs. 

## :fire: Additional features

+ Supports StyleGAN2 & StyleGAN3 on the classic attributes
+ New attributes (Bald, Gray hair, Blond hair, Earings, ...) for:
  + StyleGAN
  + StyleGAN2
  + StyleGAN3
+ Supports face generation using StyleGAN3 & StyleGAN2

The list of new features can be found on our [attributes detection classifier repository](https://github.com/clementapa/CelebFaces_Attributes_Classification/blob/main/utils/constant.py)

## :hammer: Training an attribute detection classifier

We use a ViT-base model to train an attribute detection classifier, please refer to our [classification code](https://github.com/clementapa/CelebFaces_Attributes_Classification) if you want to test it for new models. Once you retrieve the trained SVM from this repo, you can directly move them in this repo and use them.

## :star: Generate images using StyleGAN & StyleGAN2 & StyleGAN3

We did not changed anything to the structure of the old repository, please refer to the [previous README](https://github.com/younesbelkada/interfacegan/blob/master/README_old.md). For StyleGAN

### :movie_camera: Get the pretrained StyleGAN

We use the styleGAN trained on ffhq for our experiments, if you want to reproduce them, run:
```
wget -P interfacegan/models/pretrain https://www.dropbox.com/s/qyv37eaobnow7fu/stylegan_ffhq.pth
```

### :movie_camera: Get the pretrained StyleGAN2

We use the styleGAN2 trained on ffhq for our experiments, if you want to reproduce them, run:
```
wget -P models/pretrain https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-1024x1024.pkl 
```

### :movie_camera: Get the pretrained StyleGAN3

We use the styleGAN3 trained on ffhq for our experiments, if you want to reproduce them, run:
```
wget -P models/pretrain https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhq-1024x1024.pkl 
```

The pretrained model should be copied at ``` models/pretrain ```. If not, move the pretrained model file at this directory.

### :art: Run the generation script

If you want to generate 10 images using styleGAN3 downloaded before, run:
```
python generate_data.py -m stylegan3_ffhq -o output_stylegan3 -n 10
```
The arguments are exactly the same as the arguments from the original repository, the code supports the flag ```-m stylegan3_ffhq``` for styleGAN3 and ```-m stylegan3_ffhq``` for styleGAN2.

## :pencil2: Edit generated images

You can edit the generated images using our trained boundaries! Depending on the generator you want to use, make sure that you have downloaded the right model and put them into ``` models/pretrain ```. 

### Examples

Please refer to our interactive google colab notebook to play with our models by clicking the following badge:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/younesbelkada/interfacegan/blob/master/InterFaceGAN%2B%2B.ipynb)


#### StyleGAN

Example of generated images using StyleGAN and moving the images towards the direction of the attribute **grey hair**:

<p float="center">
  <img src="images/sg_before.jpeg" alt="original images generated with StyleGAN"/>
</p>
<p float="center">
  <img src="images/sg_grey_hair.jpeg" alt="grey hair version of the images generated with StyleGAN"/>
</p>

#### StyleGAN2

Example of generated images using StyleGAN2 and moving the images towards the opposite direction of the attribute **young**:

<p float="center">
  <img src="images/sg2.jpeg" alt="original images generated with StyleGAN2"/>
</p>
<p float="center">
  <img src="images/sg2_not_young.jpeg" alt="non young version of the images generated with StyleGAN2"/>
</p>

#### StyleGAN3

Example of generated images using StyleGAN3 and moving the images towards the attribute **beard**:

<p float="center">
  <img src="images/sg3_before.jpeg"/>
</p>
<p float="center">
  <img src="images/sg3_beard.jpeg"/>
</p>