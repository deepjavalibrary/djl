# DJL - core API

## Overview

This module contains the core API of the Deep Java Library (DJL) project. It includes the following packages:

- engine - Contains classes to load a deep learning engine
- inference - Contains classes to implement inference tasks
- metric - Contains classes to collect metrics information
- modality - Contains utility classes for each of the predefined modalities
- ndarray - Contains classes and interfaces to define an n-dimensional array
- nn - Contains classes to define neural network operations
- training - Contains classes to implement training tasks
- translate - Contains classes and interfaces to translate between java objects and NDArrays

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/api/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.

## Installation
You can pull the DJL API from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>api</artifactId>
    <version>0.12.0</version>
</dependency>
```

For testing the current nightly build, use the following: 

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>api</artifactId>
    <version>0.13.0-SNAPSHOT</version>
</dependency>
```

Note that the nightly build is under active development and might contain a number of bugs and 
instabilities. 
