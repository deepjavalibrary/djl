Jupyter notebook examples
=========================

## Overview

This folder contains jupyter notebooks that demonstrate how to use Joule API in real world examples. 

## Setup

### JDK 9 (not jre)

JDK 9 (or above are required)

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




