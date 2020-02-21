# DJL - fastText engine implementation

## Overview

This module contains the fastText implementation of the Deep Java Library (DJL) EngineProvider.

This is a shallow wrapper around [JFastText](https://github.com/vinhkhuc/JFastText). It has following limiations:

- fasttext-engine doesn't support NDArray operations
- Training dataset must comply with fastText format
- Saved model is fastText specific, can not be used by other engine
- fasttext-engine doesn't support building model with neural network blocks

## Why DJL over JFastText

- DJL provides unified training and inference API. This makes it easy for user to migrate to deep learning base model in future.
- Object oriented API design make it easy for java developer.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.djl.ai/fasttext-engine/0.3.0/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation
You can pull the fastText engine from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.fasttext</groupId>
    <artifactId>fasttext-engine</artifactId>
    <version>0.3.0</version>
    <scope>runtime</scope>
</dependency>
```
