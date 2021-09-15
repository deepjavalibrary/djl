# DJL - TensorFlow model zoo

The TensorFlow model zoo contains symbolic models that can be used for inference.
All the models in this model zoo contain pre-trained parameters for their specific datasets.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.tensorflow/tensorflow-model-zoo/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.

## Installation
You can pull the [ai.djl.tensorflow:tensorFlow-model-zoo](https://search.maven.org/artifact/ai.djl.tensorflow/tensorflow-model-zoo)
from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-model-zoo</artifactId>
    <version>0.12.0</version>
</dependency>
```

## Pre-trained models

The TensorFlow model zoo contains Computer Vision (CV) models.

* CV
  * Image Classification
  * Object Detection

### How to find a pre-trained model in model zoo

Please see [DJL Model Zoo](../../model-zoo/README.md)
