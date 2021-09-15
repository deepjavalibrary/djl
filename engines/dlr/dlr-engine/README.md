# DJL - DLR engine implementation(Experimental)

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

- ai.djl.dlr:dlr-engine:0.12.0

```xml
<dependency>
    <groupId>ai.djl.dlr</groupId>
    <artifactId>dlr-engine</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `dlr-engine` library, you may also need to include the Neo DLR native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

```xml
<dependency>
    <groupId>ai.djl.dlr</groupId>
    <artifactId>dlr-native-auto</artifactId>
    <version>1.6.0</version>
    <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.dlr:dlr-native-cpu:1.6.0:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.dlr</groupId>
    <artifactId>dlr-native-cpu</artifactId>
    <version>1.6.0</version>
    <scope>runtime</scope>
    <classifier>osx-x86_64</classifier>
</dependency>
```

### Linux
For Linux, you can use the following library:

- ai.djl.dlr:dlr-native-cpu:1.6.0:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.dlr</groupId>
    <artifactId>dlr-native-cpu</artifactId>
    <version>1.6.0</version>
    <scope>runtime</scope>
    <classifier>linux-x86_64</classifier>
</dependency>
```

## Load your own custom dlr
You can use environment variable to specify your custom dlr by

```
export DLR_LIBRARY_PATH=path/to/your/dlr
```

## Platform Limitation
DLR engine is still under development. The supported platform are limited to Macosx, Linux CPU. If you would like to use other platforms, please let us know.

# Multi-threading Capabilities
TVM runtime itself doesn't support multi-threading. As a result, when creating a new Predictor, we will copy the tvm model to avoid sharing the states.
We are still actively testing multithreading capability.
