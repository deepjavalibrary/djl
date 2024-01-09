# DJL - LightGBM engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for LightGBM.

It is based off the [LightGBM project](https://github.com/microsoft/LightGBM).

The package DJL delivered only contains the core inference capability.

We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the engine and make switching between engines difficult.

LightGBM is an ML library with limited support for NDArray operations.
Due to the engine's limitation, it only covers the basic NDArray creation methods.
User can only create two-dimension NDArray to form as the input.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.ml.lightgbm/lightgbm/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

#### System Requirements

LightGBM can only run on top of the Linux/Mac/Windows machine using x86_64.

## Installation
You can pull the LightGBM engine from the central Maven repository by including the following dependency:

- ai.djl.ml.lightgbm:lightgbm:0.26.0

```xml
<dependency>
    <groupId>ai.djl.ml.lightgbm</groupId>
    <artifactId>lightgbm</artifactId>
    <version>0.26.0</version>
    <scope>runtime</scope>
</dependency>
```

