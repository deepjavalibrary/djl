# NLP support with fastText

## Overview

This module contains the NLP support with fastText implementation.

fastText module's implementation in DJL is not considered as an Engine, it doesn't support Trainer and Predictor.
The training and inference functionality is directly provided through [FtModel](https://javadoc.io/doc/ai.djl.fasttext/fasttext-engine/latest/ai/djl/fasttext/FtModel.html)
class. You can find examples [here](https://github.com/deepjavalibrary/djl/blob/master/extensions/fasttext/src/test/java/ai/djl/fasttext/CookingStackExchangeTest.java).

Current implementation has the following limitations:

- Training dataset must comply with fastText format
- Saved model is fastText specific, can only be loaded with this module

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
    <version>0.12.0</version>
</dependency>
```

