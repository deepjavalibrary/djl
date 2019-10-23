Jupyter notebook examples
=========================

## Overview

This folder contains jupyter notebooks that demonstrate how to use djl.ai API in real world examples. 

## Setup

### JDK 9 (not jre)

JDK 9 (or above are required), the examples provided in this folder requires JDK 10+.

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
git clone https://github.com/SpencerPark/IJava.git
cd IJava/
chmod u+x gradlew
./gradlew installKernel
```

## Start jupyter notebook

```bash
jupyter notebook
```

## Docker setup

You may use Jupyter docker to have the same experience

### Create docker image

```
docker build -t djl.ai .
```

### Run docker image

```
docker run -p 8888:8888 -it djl.ai
```

You can open the `http://localhost:8888` to see the hosted instance on docker. 
Passed in the tokens provided in the docker message, and the docker is ready to go.