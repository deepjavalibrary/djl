# DJL - examples

This module contains examples to demonstrate use of the Deep Java Library (DJL).

The following examples are included:

- [Image classification example](docs/image_classification.md)
- [Single-shot Object detection example](docs/object_detection.md)
- [Bert question and answer example](docs/BERT_question_and_answer.md)

## Prerequisites

* You need to have Java Development Kit version 8 or later installed on your system. For more information, see [Setup](../docs/development/setup.md).
* You should be familiar with the API documentation in the DJL [Javadoc](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/api/index.html).


# Getting started: 30 seconds to run an example

## Building with the command line

This example supports building with both Gradle and Maven. To build, use either of the following commands:

### Gradle build

```sh
cd examples
./gradlew jar
```

### Maven build

```sh
cd examples
mvn package
```

### Run example code
With the gradle `application` plugin you can execute example code directly.
For more information on running each example, see the example's documentation.

The following command executes an object detection example:

```sh
cd examples
./gradlew run
```

## Engine selection

DJL is engine agnostic, so it's capable of supporting different backends. Only
the MXNet engine backend implementation is currently supported.

With MXNet, you can choose different versions of the native MXNet library.
The supplied examples use `mxnet-native-mkl` for macOS. You may need to 
change the MXNet library version to match your platform in [pom.xml](pom.xml) or [build.gradle](build.gradle).

The following MXNet versions are available:
* mxnet-native-mkl
* mxnet-native-cu101mkl
