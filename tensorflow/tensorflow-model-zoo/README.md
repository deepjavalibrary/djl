# DJL - TensorFlow model zoo

The TensorFlow model zoo contains symbolic models that can be used for inference.
All the models in this model zoo contain pre-trained parameters for their specific datasets.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.tensorflow/tensorflow-model-zoo/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.

## Installation
To use the experimental TensorFlow model zoo, you need to build from source. 
Simply begin by checking out the code.
Once you have checked out the code locally, you can build it as follows using Gradle:

```sh
./gradlew build
```

Follow the main [README.md](../../README.md) and the [quick start guide](../../docs/quick_start.md)

## Pre-trained models

The TensorFlow model zoo contains Computer Vision (CV) models. We currently only support image classification models.

* CV
  * Image Classification

### How to find a pre-trained model in model zoo

Please see [DJL Model Zoo](../../model-zoo/README.md)
