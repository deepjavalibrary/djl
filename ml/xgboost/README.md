# DJL - XGBoost engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for XGBoost.

It is based off the [DMLC: XGBoost](https://github.com/dmlc/xgboost).

The package DJL delivered only contains the core inference capability. All Scala and Hadoop dependencies are removed
from the original distribution. The package is really light-weight to be deployed.

We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the XGBoost and make switching between engines difficult.

XGBoost is a ML library with limited support for NDArray operations.
Due to the engine's limitation, it only covers the basic NDArray creation methods.
User can only create two-dimension NDArray to form as the input.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.onnxruntime/onnxruntime-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

#### System Requirements

XGBoost can only run on top of the Linux/Mac machine. User can build from source to provide `xgboost4j.dll` for Windows.

## Installation
You can pull the XGBoost engine from the central Maven repository by including the following dependency:

- ai.djl.ml.xgboost:xgboost:0.12.0

```xml
<dependency>
    <groupId>ai.djl.ml.xgboost</groupId>
    <artifactId>xgboost</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```

