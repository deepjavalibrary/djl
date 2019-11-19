# DJL - Basic Dataset

## Overview

This module contains the basic datasets of the Deep Java Library's (DJL) project. These datasets are used to train deep learning models.

## List of datasets

This module contains the following datasets:

- [MNIST](http://yann.lecun.com/exdb/mnist/) - A handwritten digits dataset
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) - A dataset consisting of 60,000 32x32 color images in 10 classes
- [Coco](http://cocodataset.org) - A large-scale object detection, segmentation, and captioning dataset that contains 1.5 million object instances 
- [ImageNet](http://www.image-net.org/) - An image database organized according to the WordNet hierarchy
  >**Note**: You have to manually download the ImageNet dataset due to licensing requirements.
- Pikachu - 1000 Pikachu images of different angles and sizes created using an open source 3D Pikachu model
- ImageFolder - A custom dataset that can get images from a local folder

## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>basicdataset</artifactId>
    <version>0.2.0</version>
</dependency>
```
