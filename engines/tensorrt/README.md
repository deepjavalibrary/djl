# DJL - TensorRT engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for TensorRT.

It is based off the [TensorRT](https://github.com/NVIDIA/TensorRT).

We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the TensorRT and make switching between engines difficult.

TensorRT is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine](../../docs/hybrid_engine.md).

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.tensorrt/tensorrt/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux:
./gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

## Installation
You can pull the TensorRT engine from the central Maven repository by including the following dependency:

- ai.djl.tensorrt:tensorrt:0.28.0

```xml
<dependency>
    <groupId>ai.djl.tensorrt</groupId>
    <artifactId>tensorrt</artifactId>
    <version>0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

## Development
We provide a [docker file](https://github.com/deepjavalibrary/djl/blob/master/docker/tensorrt/Dockerfile) to make 
development of tensorrt with djl easier. Follow the instructions in the 
[docker readme](https://github.com/deepjavalibrary/djl/blob/master/docker/README.md) to build and run the container.
