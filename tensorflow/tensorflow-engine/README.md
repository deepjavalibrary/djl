# DJL - Tensorflow engine implementation

## Overview

This module contains the Tensorflow implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend that developers use classes in this module directly. Use of these classes will couple your code with TensorFlow and make switching between frameworks difficult. Even so, developers are not restricted from using engine-specific features. For more information, see [NDManager#invoke()](https://javadoc.io/static/ai.djl/api/0.5.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

**Currently Windows platform and training is not supported.**

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.tensorflow/tensorflow-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation

To use the experimental TensorFlow engine, you need to build from source. 
Simply begin by checking out the code.
Once you have checked out the code locally, you can build it as follows using Gradle:

```sh
./gradlew build
```

Follow the main [README.md](../../README.md) and the [quick start guide](../../docs/quick_start.md)
