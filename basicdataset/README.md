# DJL - Basic Dataset

## Overview

This module contains a number of basic and standard datasets in the Deep Java Library's (DJL). These datasets are used to train deep learning models.

You can find the datasets provided by this module on our [docs](http://docs.djl.ai/docs/dataset.html).

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl/basicdataset/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\gradlew javadoc
```
The javadocs output is built in the build/doc/javadoc folder.


## Installation
You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl</groupId>
    <artifactId>basicdataset</artifactId>
    <version>0.24.0</version>
</dependency>
```

Some datasets(e.g. COCO) contains non-standard image files. OpenJDK may fail to load these images.
twelvemonkeys ImageIO plugins provide a wide range of image format support. If you need to load
images that not supported by default JDK, you can consider add the following dependencies into your project:

```xml
    <dependency>
        <groupId>com.twelvemonkeys.imageio</groupId>
        <artifactId>imageio-jpeg</artifactId>
        <version>3.8.3</version>
    </dependency>
    <dependency>
        <groupId>com.twelvemonkeys.imageio</groupId>
        <artifactId>imageio-bmp</artifactId>
        <version>3.8.3</version>
    </dependency>
    ...
```
