# DJL - TensorFlow engine implementation

## Overview

This module contains the Deep Java Library (DJL) EngineProvider for TensorFlow.

We don't recommend that developers use classes in this module directly. Use of these classes will
couple your code with TensorFlow and make switching between frameworks difficult.

**Currently training is not supported.**

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.tensorflow/tensorflow-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation
You can pull the TensorFlow engine from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-engine</artifactId>
    <version>0.10.0</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `tensorflow-engine` library, you may also need to include the TensorFlow native library in your project.

### Install TensorFlow native library

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-auto</artifactId>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
```
