# Audio support with ffmpeg

This module contains the audio support extension with [JavaCV](https://github.com/bytedeco/javacv).

Right now, the package provides an `SpeechRecognitionDataset` that allows you extract features from audio file.

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
    <groupId>ai.djl.audio</groupId>
    <artifactId>audio</artifactId>
    <version>0.19.0</version>
</dependency>
```
