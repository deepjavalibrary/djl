# Image support with OpenCV

This module contains the image support extension with [OpenCV](https://opencv.org/). It is based on the [Java package from OpenPnP](https://github.com/openpnp/opencv).

Right now, the package provides an `OpenCVImage` that acts as a faster implementation than the native `BufferedImage`. Once this package is added to your classpath, it will automatically be used through the standard DJL `ImageFactory`.

## Documentation

You can build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation

You can pull the module from the central Maven repository by including the following dependency in your `pom.xml` file:

```xml
<dependency>
    <groupId>ai.djl.opencv</groupId>
    <artifactId>opencv</artifactId>
    <version>0.19.0</version>
</dependency>
```
