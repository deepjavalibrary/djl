# NLP support with fastText

## Overview

This module contains the NLP support with fastText implementation.

This is a shallow wrapper around [JFastText](https://github.com/vinhkhuc/JFastText). It has following limitations:

- Training dataset must comply with fastText format
- Saved model is fastText specific, can only be loaded with this module
- fastText model is not a full implementation of DJL Model, it doesn't support Trainer and Predictor.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.fasttext/fasttext-engine/latest/index.html).

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
    <version>0.6.0</version>
</dependency>
```
ß
