# Streaming support with RxJava

This module contains the image support extension with [RxJava](https://github.com/ReactiveX/RxJava).

Right now, the package provides a `StreamingBlock` that adds support for streamable blocks (must return list data) and an equivalent `StreamingPredictor`.

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.rx/rx/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation

You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.rx</groupId>
    <artifactId>rx</artifactId>
    <version>0.21.0</version>
</dependency>
```
