# NLP support with fastText

## Overview

This module contains the NLP support with fastText implementation.

fastText module's implementation in DJL is not considered as an Engine, it doesn't support Trainer and Predictor.
Training is only supported by using [TrainFastText](https://javadoc.io/doc/ai.djl.fasttext/fasttext-engine/latest/ai/djl/fasttext/TrainFastText.html).
This produces a special block which can perform inference on its own or by using a model and predictor.
Pre-trained FastText models can also be loaded by using the standard DJL criteria.
You can find examples [here](https://github.com/deepjavalibrary/djl/blob/master/extensions/fasttext/src/test/java/ai/djl/fasttext/CookingStackExchangeTest.java).

Current implementation has the following limitations:

- Training dataset must comply with fastText format
- Saved model is fastText specific, can only be loaded with this module

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.fasttext/fasttext-engine/latest/index.html).

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
    <version>0.17.0</version>
</dependency>
```

