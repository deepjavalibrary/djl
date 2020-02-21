# DJL - examples

This module contains examples to demonstrate use of the Deep Java Library (DJL).

The following examples are included:

- [Single-shot object detection example](docs/object_detection.md)
- [Train your first model](docs/train_mnist_mlp.md)
- [Image classification example](docs/image_classification.md)
- [Transfer learning example](docs/train_cifar10_resnet.md)
- [Train SSD model example](docs/train_pikachu_ssd.md)
- [Multi-threaded inference example](docs/multithread_inference.md)
- [Bert question and answer example](docs/BERT_question_and_answer.md)
- [Instance segmentation example](docs/instance_segmentation.md)
- [Pose estimation example](docs/pose_estimation.md)
- [Action recognition example](docs/action_recognition.md)
- [Multi-label dataset training example](docs/train_captcha.md)

## Prerequisites

* You need to have Java Development Kit version 8 or later installed on your system. For more information, see [Setup](../docs/development/setup.md).
* You should be familiar with the API documentation in the DJL [Javadoc](https://javadoc.djl.ai/api/0.3.0/index.html).


# Getting started: 30 seconds to run an example

## Building with the command line

This example supports building with both Gradle and Maven. To build, use either of the following commands:

* Gradle build
    ```sh
    cd examples
    ./gradlew jar
    ```

* Maven build
    ```sh
    cd examples
    mvn package -DskipTests
    ```

### Run example code
With the gradle `application` plugin you can execute example code directly.
For more information on running each example, see the example's documentation.

The following command executes an object detection example:

* Gradle
    ```sh
    cd examples
    ./gradlew run
    ```

* Maven
    ```sh
    cd examples
    mvn package -DskipTests
    mvn exec:java -Dexec.mainClass="ai.djl.examples.inference.ObjectDetection"
    ```

## Engine selection

DJL is engine agnostic, so it's capable of supporting different backends. Only
the MXNet engine backend implementation is currently supported.

With MXNet, you can choose different builds of the native MXNet library.
We recommend the automatic engine selection which downloads the best engine for your platform and available hardware during the first runtime.
Activate the automatic selection by adding `ai.djl.mxnet:mxnet-native-auto:1.6.0-SNAPSHOT` as a dependency.
You can also see [MXNet Engine](../mxnet/mxnet-engine/README.md) for further options.
