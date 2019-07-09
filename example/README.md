Joule examples
==============

This module contains example project to demonstrate how developer can use Joule API.

There are three examples:

1. Image classification example
2. Single-shot Object detection example
3. Bert question ans answer example

Getting started: 30 seconds to run an example
=======================

## Import the Joule with Intellij

### Gradle

1. Open Intellij and click `Import Project`.
2. Find the `build.gradle` directly in Joule folder.
Note that there are 7 build.gradle in api/example/integration/jnarator/mxnet/native/tensorflow seperately,
make sure to select the one only one layer below Joule.
3. Use the default configuration and click `OK`.
4. Please go to seperate example to continue.
[Image classification example](CLASSIFY.md)
[Single-shot Object detection example]()
[Bert question ans answer example](BERTQA.md)

---
## Building From Source

If you want to build the example from source, you can build it using gradle once you check out the code.

```sh
cd examples
./gradlew build
```

If you want to skip unit test:
```sh
./gradlew build -x test
```

By default, Joule examples will use `mxnet-mkl` as a backend.

Available mxnet versions are as follows:

| Version  |
| -------- |
| mxnet-mkl|
| mxnet-cu101mkl|

---
Please find more information here:
1. [Javadoc](https://joule.s3.amazonaws.com/java-api/index.html)






