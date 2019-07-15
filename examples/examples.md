Joule examples
==============

This module contains example projects to demonstrate how developers can use the Joule API.

There are three examples in this project:

1. [Image classification example](CLASSIFY.md)
2. [Single-shot Object detection example](SSD.md)
3. [Bert question and answer example](BERTQA.md)

## Prerequisite

You need to have JDK 8 (or later) and IntelliJ installed on your system. Read [here](setup.md) for more detail.
 
### Joule API reference
You should also be familiar with the Joule API documentation: [Javadoc](https://joule.s3.amazonaws.com/java-api/index.html)


Getting started: 30 seconds to run an example
=======================

## Building with command line

This example project supports building with both gradle and maven. To build, use the following:

### gradle

```sh
cd examples
./gradlew build
```

### maven build

```sh
cd examples
mvn package
```

### Run example code
With the gradle `application` plugin you can execute example code directly.
You can find more detail in each example's detail document.
Here is an example that executes classification:

```sh
cd examples
./gradlew run --args="-n squeezenet_v1.1 -i ./src/test/resources/kitten.jpg"
```

## Import Joule example project with IntelliJ

1. Open IntelliJ and click `Import Project`.
2. Select the `examples` directory in the Joule folder, and click "Open".
3. Choose `Import project form existing model`, you can select either `Gradle` or `Maven`
4. Use the default configuration and click `OK`.
5. Select an example to continue.

## Engine selection

Joule is engine agnostic, so you can choose different engine providers. We currently
provide MXNet engine implementation.

With MXNet, you can choose different flavors of the native MXNet library.
In this example, we use `mxnet-native-mkl` for OSX platform. You might need to 
change it for your platform in [pom.xml](pom.xml) or [build.gradle](build.gradle).


Available MXNet versions are as follows:

| Version  |
| -------- |
| mxnet-mkl|
| mxnet-cu101mkl|

