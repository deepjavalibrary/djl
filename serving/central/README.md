# DJL - Central Server

## Overview

This module contains the DJLCentral web interface implementation.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/serving/central/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation
You can pull the server from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.serving</groupId>
    <artifactId>central</artifactId>
    <version>0.12.0</version>
    <scope>runtime</scope>
</dependency>
```

## Run model server

Use the following command to start model server locally:

```sh
cd serving/central

# for Linux/macOS:
./gradlew run

# for Windows:
..\..\gradlew run
```

The DJLCentral server will be listening on port 8080.

open your browser an type in url 

```ssh
http://localhost:8080/
```
