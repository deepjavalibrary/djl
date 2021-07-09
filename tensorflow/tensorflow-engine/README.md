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

**Note:** One of the dependency *JavaCPP* has a bug that will cause memory leak. See
[here](https://github.com/bytedeco/javacpp/commit/7f27899578dfa18e22738a3dd49701e1806b464a) for
more detail. The issue has been fixed in javacpp 1.5.6-SNAPSHOT version. You need to include the
javacpp SNAPSHOT version explicitly to avoid memory leak:

```
runtimeOnly "org.bytedeco:javacpp:1.5.6-SNAPSHOT"
runtimeOnly ("ai.djl.tensorflow:tensorflow-engine") {
    exclude group: "org.bytedeco", module: "javacpp"
}
```

You can pull the TensorFlow engine from the central Maven repository by including the following dependency:

- ai.djl.tensorflow:tensorflow-engine:0.12.0

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-engine</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```
Besides the `tensorflow-engine` library, you may also need to include the TensorFlow native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

- ai.djl.tensorflow:tensorflow-native-auto:2.4.1

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-auto</artifactId>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.tensorflow:tensorflow-native-cpu:2.4.1:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

- ai.djl.tensorflow:tensorflow-native-cu110:2.4.1:linux-x86_64 - CUDA 11.0

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cu110</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

### Linux CPU

- ai.djl.tensorflow:tensorflow-native-cpu:2.4.1:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.4.1</version>
</dependency>
```

### Windows

For the Windows platform, you can choose between CPU and GPU.

#### Windows GPU

- ai.djl.tensorflow:tensorflow-native-cu110:2.4.1:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cu110</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.4.1</version>
    <scope>runtime</scope>
</dependency>
```

### Windows CPU

- ai.djl.tensorflow:tensorflow-native-cpu:2.4.1:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.4.1/version>
</dependency>
```
