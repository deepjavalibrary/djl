# DJL - DLR engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for DLR.

It is based off the [Neo DLR](https://github.com/neo-ai/neo-ai-dlr).


We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the DLR and make switching between engines difficult.

DLR is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine for ONNX Runtime and DLR](../../docs/hybrid_engine.md).

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.dlr/dlr-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

## Installation
You can pull the DLR engine from the central Maven repository by including the following dependency:

- ai.djl.dlr:dlr-engine:0.9.0-SNAPSHOT

```xml
<dependency>
    <groupId>ai.djl.dlr</groupId>
    <artifactId>dlr-engine</artifactId>
    <version>0.9.0-SNAPSHOT</version>
    <scope>runtime</scope>
</dependency>
```

## Load your own custom dlr
You can use environment variable to specify your custom dlr by
```
export DLR_LIBRARY_PATH=path/to/your/dlr
```

## Limitation
DLR engine is still under development. The supported platform are limited to Macosx, Linux CPU and we are still actively testing multithreading capability.
