# DJL - PaddlePaddle engine implementation

## Overview

This module contains the Deep Java Library (DJL) EngineProvider for PaddlePaddle.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with PaddlePaddle and make switching between frameworks difficult.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.paddlepaddle/paddlepaddle-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation
You can pull the PaddlePaddle engine from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-engine</artifactId>
    <version>0.8.0</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `paddlepaddle-engine` library, you may also need to include the PaddlePaddle native library in your project.
All current provided PaddlePaddle native libraries come from PaddlePaddle pip package:

- https://pypi.org/project/paddlepaddle/#files
- https://pypi.org/project/paddlepaddle-gpu/#files

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-auto</artifactId>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.paddlepaddle:paddlepaddle-native-mkl:1.8.5:osx-x86_64

    This package takes advantage of the Intel MKL library to boost performance.

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-mkl</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have Nvidia [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

- ai.djl.paddlepaddle:paddlepaddle-native-cu102:1.8.5:linux-x86_64 - CUDA 10.2
- ai.djl.paddlepaddle:paddlepaddle-native-cu101:1.8.5:linux-x86_64 - CUDA 10.1

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu102</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu101</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

#### Linux CPU

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:1.8.5:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>1.8.5</version>
</dependency>
```

### Windows

For the Windows platform, you can use CPU package.

#### Windows GPU

- ai.djl.paddlepaddle:paddlepaddle-native-cu102:1.8.5:win-x86_64 - CUDA 10.2
- ai.djl.paddlepaddle:paddlepaddle-native-cu101:1.8.5:win-x86_64 - CUDA 10.1

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu102</artifactId>
    <classifier>win-x86_64</classifier>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu101</artifactId>
    <classifier>win-x86_64</classifier>
    <version>1.8.5</version>
    <scope>runtime</scope>
</dependency>
```

### Windows CPU

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:1.8.5:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>1.8.5</version>
</dependency>
```
