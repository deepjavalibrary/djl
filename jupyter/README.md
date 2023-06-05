# DJL - Jupyter notebooks

## Overview

This folder contains tutorials that illustrate how to accomplish basic AI tasks with Deep Java Library (DJL).

## [Beginner Tutorial](tutorial/README.md)

## More Tutorial Notebooks

- [Run object detection with model zoo](object_detection_with_model_zoo.ipynb)
- [Load pre-trained PyTorch model](load_pytorch_model.ipynb)
- [Load pre-trained Apache MXNet model](load_mxnet_model.ipynb)
- [Transfer learning example](transfer_learning_on_cifar10.ipynb)
- [Question answering example](BERTQA.ipynb)

You can run our notebook online: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deepjavalibrary/djl/master?filepath=jupyter)

## Setup

### JDK 11 (not jre)

JDK 11 (or above are required) to run the examples provided in this folder.

to confirm the java path is configured properly:

```bash
java --list-modules | grep "jdk.jshell"

> jdk.jshell@12.0.1
```

### Install jupyter notebook on python3

```bash
pip3 install jupyter
```

### Install IJava kernel for jupyter

```bash
git clone https://github.com/frankfliu/IJava.git
cd IJava/
./gradlew installKernel
```

## Start jupyter notebook

```bash
jupyter notebook
```

## Docker setup

You may want to use docker for simple installation or you are using Windows.

### Run docker image

```sh
cd jupyter
docker run -itd -p 127.0.0.1:8888:8888 -v $PWD:/home/jupyter deepjavalibrary/jupyter
```

You can open the `http://localhost:8888` to see the hosted instance on docker.

### Build docker image by yourself

You can read [Dockerfile](https://github.com/deepjavalibrary/djl/blob/master/jupyter/Dockerfile) for detail. To build docker image:

```sh
cd jupyter
docker build -t deepjavalibrary/jupyter .
```

### Run docker compose

```sh
cd jupyter
docker-compose build
docker-compose up -d
```

You can open the `http://localhost:8888` to see the hosted instance on docker compose.
