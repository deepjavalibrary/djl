# DeepJavaLibrary - examples

This module contains examples to demonstrate how developers can use the DeepJavaLibrary API.

The following is a list of examples:

- [Image classification example](docs/image_classification.md)
- [Single-shot Object detection example](docs/object_detection.md)
- [Bert question and answer example](docs/BERT_question_and_answer.md)

## Prerequisite

* You need to have JDK 8 (or later) installed on your system. Read [here](../docs/development/setup.md) for more detail.
* You should also be familiar with the API documentation: [Javadoc](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/index.html)


# Getting started: 30 seconds to run an example

## Building with command line

This example project supports building with both gradle and maven. To build, use the following:

### gradle

```sh
cd examples
./gradlew jar
```

### maven build

```sh
cd examples
mvn package
```

### Run example code
With the gradle `application` plugin you can execute example code directly.
You can find how to run each example in each example's detail document.
Here is an example that executes object detection example:

```sh
cd examples
./gradlew run
```

## Engine selection

djl.ai is engine agnostic, so you can choose different engine providers. We currently
provide MXNet engine implementation.

With MXNet, you can choose different flavors of the native MXNet library.
In this example, we use `mxnet-native-mkl` for OSX platform. You might need to 
change it for your platform in [pom.xml](pom.xml) or [build.gradle](build.gradle).

Available MXNet versions are as follows:

| Version              |
| -------------------- |
| mxnet-native-mkl     |
| mxnet-native-cu101mkl|
