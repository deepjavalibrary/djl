# DJL Zero

## Overview

This module is a zero deep learning knowledge required wrapper over DJL.
Instead of worrying about finding a model or how to train, this will provide a simple recommendation for your deep learning application.
It is the easiest way to get started with DJL and get a solution for your deep learning problem.

DJL Zero is based on the [Application](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/Application.html) or the common deep learning task you wish to train.
It features an automatic training method (if currently implemented) and an automatic recommendation for pre-trained models (if applicable for the application).

The automatic training method is designed to be simple and requires only two parameters.
First is a `Dataset`, which all can be implemented with helpers found on our [guide to creating a dataset](../docs/development/how_to_use_dataset.md).
And second, it requires a desired `Performance`: one of `FAST`, `BALANCED`, or `ACCURATE` depending on your desired place in the model size curve.
In general, larger models are more accurate in exchange for requiring more disk space, cost, and latency.

Some applications also feature a pre-trained method.
This will pull a model for the application from the DJL model zoo that is recommended by the DJL team.
Depending on the application, there may also be options in terms of what performance level, classifications, output type, etc.

## List of Applications

This module contains the following applications:

- [Image Classification](https://javadoc.io/doc/ai.djl/djl-zero/latest/ai/djl/zero/cv/ImageClassification.html) - take an image and classify the main subject of the image.
- [Object Detection](https://javadoc.io/doc/ai.djl/djl-zero/latest/ai/djl/zero/cv/ObjectDetection.html) - take an image and find each object (car, bike, plant, etc.) in the image along with the bounding box around it's location
- [Tabular Regression](https://javadoc.io/doc/ai.djl/djl-zero/latest/ai/djl/zero/tabular/TabularRegression.html) - takes input features from a row in a table and predict the value of a numeric column

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl/zero/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>djl-zero</artifactId>
    <version>0.20.0</version>
</dependency>
```
