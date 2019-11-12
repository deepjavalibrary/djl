# DeepJavaLibrary - core API

## Overview

This module is core API of the djl.ai project. It include the following packages:

- engine - Contains classes responsible for loading a deep learning framework
- inference - Contains classes to implement inference tasks
- metric - Contains classes to collect metrics information
- modality - Contains utility classes for each of the predefined modalities
- ndarray - Contains classes and interfaces that define an n-dimensional array
- nn - Contains classes defines neural network operators
- training - Contains classes to implement training tasks
- translate - Contains classes and interfaces that translate between java objects and NDArrays

## Documentation

The Javadoc can be found [here](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/index.html).

You can build latest javadoc use the following command:

```sh
./gradlew javadoc
```
You can find generated javadoc in build/doc/javadoc folder.

## Installation
You can pull it from the central Maven repositories:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>api</artifactId>
    <version>0.2.0</version>
</dependency>
```
