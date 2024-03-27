# DJL - TensorFlow engine implementation

## Overview

This module contains the Deep Java Library (DJL) EngineProvider for TensorFlow.

We don't recommend that developers use classes in this module directly. Use of these classes will
couple your code with TensorFlow and make switching between frameworks difficult.

**Currently training is not supported.**

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.tensorflow/tensorflow-engine/latest/index.html).

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

- ai.djl.tensorflow:tensorflow-engine:0.27.0

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-engine</artifactId>
    <version>0.27.0</version>
    <scope>runtime</scope>
</dependency>
```
By default, DJL will download the TensorFlow native libraries into [cache folder](../../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

You can choose a native library based on your platform if you don't have network access at runtime.

### macOS
For macOS, you can use the following library:

- ai.djl.tensorflow:tensorflow-native-cpu:2.7.0:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.7.0</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

- ai.djl.tensorflow:tensorflow-native-cu110:2.7.0:linux-x86_64 - CUDA 11.3

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cu113</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.7.0</version>
    <scope>runtime</scope>
</dependency>
```

### Linux CPU

- ai.djl.tensorflow:tensorflow-native-cpu:2.7.0:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.7.0</version>
</dependency>
```

### Windows

For the Windows platform, you can choose between CPU and GPU.

#### Windows GPU

- ai.djl.tensorflow:tensorflow-native-cu113:2.7.0:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cu113</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.7.0</version>
    <scope>runtime</scope>
</dependency>
```

### Windows CPU

- ai.djl.tensorflow:tensorflow-native-cpu:2.7.0:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.7.0</version>
</dependency>
```
