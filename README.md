# InterfaceGAN++: Exploring the limits of InterfaceGAN

<p float="left">
  <img src="images/bald2.gif" width="200" />
  <img src="images/blond.gif" width="200" /> 
  <img src="images/makeup.gif" width="200" /> 
  <img src="images/gray_hair.gif" width="200" /> 
</p>

> From left to right - Images generated using styleGAN2 and the boundaries *Bald*, *Blond*, *Heavy_Makeup*, *Gray_Hair*

This the the repository to a project related to the *Introduction to Numerical Imaging* (*i.e, Introduction à l'Imagerie Numérique* in French), given by the MVA Masters program at ENS-Paris Saclay. The project and repository is based on the work from [Shen et al.](https://github.com/younesbelkada/interfacegan/blob/master/README_old.md), and fully supports their codebase. You can refer to the [original README](https://github.com/younesbelkada/interfacegan/blob/master/README_old.md)) to reproduce their results.

## Introduction

> In this repository, we propose an approach, termed as InterFaceGAN++, for semantic face editing based on the work from Shen et al. Specifically, we leverage the ideas from the previous work, by applying the method for new face attributes, and also for StyleGAN3. We show qualitative results of our method and we will attempt to draw some conclusion based on the observations

## :fire: Additional features

+ Supports styleGAN3 on the classic attributes
+ New attributes (Bald, Gray hair, Blond hair, Earings, ...) for:
  + StyleGAN2
  + StyleGAN3

The list of new features can be found on our [attributes detection classifier repository](https://github.com/clementapa/CelebFaces_Attributes_Classification/blob/main/utils/constant.py)

## Training an attribute detection classifier

We use a ViT-base model to train an attribute detection classifier, please refer to our classification code if you want to test it for new models. Once you retrieve the trained SVM from this repo, you can directly move them in this repo and use them.

