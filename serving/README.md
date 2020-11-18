# DJL - Model Server

## Overview

This module contains an universal model serving implementation.

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl/serving/latest/index.html).

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
    <groupId>ai.djl</groupId>
    <artifactId>serving</artifactId>
    <version>0.8.0</version>
    <scope>runtime</scope>
</dependency>
```

## Run model server

Use the following command to start model server locally:

```sh
cd serving

# for Linux/macOS:
./gradlew run

# for Windows:
..\..\gradlew run
```

The model server will be listening on port 8080.

You can also load a model for serving on start up:

```sh
./gradlew run --args="-m https://resources.djl.ai/test-models/mlp.tar.gz"
```

Open another terminal, and type the following command to test the inference REST API:

```sh
cd serving
curl -X POST http://127.0.0.1:8080/predictions/mlp -F "data=@../examples/src/test/resources/0.png"

{
  "classNames": [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
  ],
  "probabilities": [
    0.9999998807907104,
    2.026697559776025E-11,
    1.249230336952678E-7,
    2.777162111389231E-10,
    1.3042782132099973E-11,
    6.133447222333999E-11,
    7.507424681918451E-10,
    2.7874487162904416E-9,
    1.0341382195022675E-9,
    4.075440429573973E-9
  ]
}
```
