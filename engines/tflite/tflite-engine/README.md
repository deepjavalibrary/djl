# DJL - TensorFlow Lite engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for TensorFlow Lite.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with TensorFlow Lite and make switching between frameworks difficult.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.tflite/tflite-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation
You can pull the TensorFlow Lite engine from the central Maven repository by including the following dependency:

- ai.djl.tflite:tflite-engine:0.25.0

```xml
<dependency>
    <groupId>ai.djl.tflite</groupId>
    <artifactId>tflite-engine</artifactId>
    <version>0.25.0</version>
    <scope>runtime</scope>
</dependency>
```

By default, DJL will download the TensorFlow Lite native libraries into [cache folder](../../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

You can choose a native library based on your platform if you don't have network access at runtime.

### macOS
For macOS, you can use the following library:

- ai.djl.tflite:tflite-native-cpu:2.4.1:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.tflite</groupId>
    <artifactId>tflite-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can use the following library:

- ai.djl.tflite:tflite-native-cpu:2.4.1:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.tflite</groupId>
    <artifactId>tflite-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.4.1</version>
</dependency>
```
