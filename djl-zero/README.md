# DJL Zero

## Overview

This module is a zero deep learning knowledge required wrapper over DJL. Instead of worrying about finding a model or how to train, this will provide a simple recommendation for your deep learning application. It is the easiest way to get started with DJL and get a solution for your deep learning problem.

## List of Applications

This module contains the following applications:

- Image Classification - take an image and classify the main subject of the image.


## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/zero/latest/index.html).

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
    <artifactId>djl-zero</artifactId>
    <version>0.12.0</version>
</dependency>
```
