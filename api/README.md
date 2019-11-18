# DJL - core API

## Overview

This module contains the core API of the Deep Java Library (DJL) project. It includes the following packages:

- engine - Contains classes responsible for loading a deep learning framework
- inference - Contains classes for implementing inference tasks
- metric - Contains classes to collect metrics information
- modality - Contains utility classes for each of the predefined modalities
- ndarray - Contains classes and interfaces that define an n-dimensional array
- nn - Contains classes that define neural network operators
- training - Contains classes to implement training tasks
- translate - Contains classes and interfaces that translate between java objects and NDArrays

## Documentation

The Javadoc can be found [here](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/api/index.html).

You can build the latest javadoc locally using the following command:

```sh
./gradlew javadoc
```
The generated javadoc will be built in the build/doc/javadoc folder.

## Installation
You can pull the DJL API from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>api</artifactId>
    <version>0.2.0</version>
</dependency>
```
