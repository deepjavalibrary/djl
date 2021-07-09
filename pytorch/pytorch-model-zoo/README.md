# DJL - PyTorch model zoo

The PyTorch model zoo contains symbolic (JIT Traced) models that can be used for inference.
All the models in this model zoo contain pre-trained parameters for their specific datasets.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.pytorch/pytorch-model-zoo/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.

## Installation
You can pull the PyTorch engine from the central Maven repository by including the following dependency in you `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-model-zoo</artifactId>
    <version>0.12.0</version>
</dependency>
```

## Pre-trained models

The PyTorch model zoo contains Computer Vision (CV) models. All the models are grouped by task under these two categories as follows:

* CV
  * Image Classification
  * Object Detection

### How to find a pre-trained model in model zoo

Please see [DJL Model Zoo](../../model-zoo/README.md)
