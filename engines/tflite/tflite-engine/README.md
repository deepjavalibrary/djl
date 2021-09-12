# DJL - TensorFlow Lite engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for TensorFlow Lite.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with TensorFlow Lite and make switching between frameworks difficult.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.tflite/tflite-engine/latest/index.html).

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

- ai.djl.tflite:tflite-engine:0.12.0

```xml
<dependency>
    <groupId>ai.djl.tflite</groupId>
    <artifactId>tflite-engine</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```
Besides the `tflite-engine` library, you may also need to include the TensorFlow Lite native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform.

- ai.djl.tflite:tflite-native-auto:2.4.1

```xml
<dependency>
    <groupId>ai.djl.tflite</groupId>
    <artifactId>tflite-native-auto</artifactId>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

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
