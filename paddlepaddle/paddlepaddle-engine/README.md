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
    <version>0.10.0-SNAPSHOT</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `paddlepaddle-engine` library, you may also need to include the PaddlePaddle native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

```xml
<dependency>
    <groupId>ai.djl.paddlepaddle</groupId>
    <artifactId>paddlepaddle-native-auto</artifactId>
    <version>2.0.0-SNAPSHOT</version>
    <scope>runtime</scope>
</dependency>
```

### Mac/Windows/Linux standalone native jarss

Currently, PaddlePaddle 2.0.0 is still under development, we will update here once it is released. You may use the auto jar
to have 2.0.0-rc1 releases for testing.
