# DJL - PaddlePaddle engine implementation

## Overview

This module contains the Deep Java Library (DJL) EngineProvider for PaddlePaddle.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with PaddlePaddle and make switching between frameworks difficult.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.paddlepaddle/paddlepaddle-engine/latest/index.html).

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
    <version>0.27.0</version>
    <scope>runtime</scope>
</dependency>
```

By default, DJL will download the PaddlePaddle native libraries into [cache folder](../../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

You can choose a native library based on your platform if you don't have network access at runtime.

### macOS
For macOS, you can use the following library:

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.2.2:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```

### Linux

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.2.2:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```

#### Linux GPU

To use Linux packages, users are also required to set `LD_LIBRARY_PATH` to the folder:

```sh
LD_LIBRARY_PATH=$HOME/.djl.ai/paddle/2.2.2-<cuda-flavor>-linux-x86_64
```

- ai.djl.paddlepaddle:paddlepaddle-native-cu102:2.2.2:linux-x86_64 - CUDA 10.2
- ai.djl.paddlepaddle:paddlepaddle-native-cu112:2.2.2:linux-x86_64 - CUDA 11.2

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu102</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu112</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```


### Windows

- ai.djl.paddlepaddle:paddlepaddle-native-cpu:2.2.2:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```

#### Windows GPU Experimental

- ai.djl.paddlepaddle:paddlepaddle-native-cu110:2.2.2:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-cu110</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.2.2</version>
    <scope>runtime</scope>
</dependency>
```
