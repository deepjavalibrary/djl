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
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `paddlepaddle-engine` library, you may also need to include the PaddlePaddle native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

- ai.djl.paddlepaddle:paddlepaddle-native-auto:2.0.2

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-auto</artifactId>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.0.2:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```

### Linux

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.0.2:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```

#### Linux GPU

To use Linux packages, users are also required to set `LD_LIBRARY_PATH` to the folder:

```
LD_LIBRARY_PATH=$HOME/.djl.ai/paddle/2.0.2-<cuda-flavor>-linux-x86_64
```

- ai.djl.paddlepaddle:paddlepaddle-native-cu101:2.0.2:linux-x86_64 - CUDA 10.1
- ai.djl.paddlepaddle:paddlepaddle-native-cu102:2.0.2:linux-x86_64 - CUDA 10.2

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu101</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu102</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```


### Windows

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.0.2:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.0.2</version>
    <scope>runtime</scope>
</dependency>
```
