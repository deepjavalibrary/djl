# Development Guideline

## Introduction

Thank you for your interest in contributing to the Deep Java Library (DJL).
In this document, we will cover everything you need to build, test, and debug your code when developing DJL.

Many of us use the [IntelliJ IDEA IDE](https://www.jetbrains.com/idea/) to develop DJL and we will sometimes mention it. However, there is no requirement to use this IDE.

## Build

This project uses a gradle wrapper, so you don't have to install gradle on your machine. You can just call the gradle wrapper using the following command:
```
./gradlew
```

There are several gradle build targets you can use. The following are the most common targets:

- `formatJava` or `fJ`: clean up and reformat your Java code
- `build`: build the whole project and run all tests
- `javadoc`: build the javadoc only
- `jar`: build the project only

You can also run this from a subfolder to build for only the module within that folder.

Run the following command to list all available tasks:
```sh
./gradlew tasks --all
```

## Test

Sometimes you may need to run individual tests or examples.
If you are developing with an IDE, you can run a test by selecting the test and clicking the "Run" button.

From the command line, you can run the following command to run a test:
```
./gradlew :<module>:run -Dmain=<class_name> --args ""
```
For example, if you would like to run the integration test, you can use the following command:
```
./gradlew :integration:run -Dmain=ai.djl.integration.IntegrationTest
```

## Logging

To get a better understanding of your problems when developing, you can enable logging by adding the following parameter to your test command:
```
-Dai.djl.logging.level=debug
```

The values to set the log level to can be found [here](https://logging.apache.org/log4j/2.x/manual/customloglevels.html).

## Debug

When debugging a DJL application in IntelliJ, it is often helpful to inspect your NDArray variables. Because NDArrays may contain
a large number of elements, rendering them can be resource-heavy and cause the debugger to hang.

IntelliJ allows you to [customize the data view in the Debug Tools](https://www.jetbrains.com/help/idea/customize-data-views.html).
You can create your own NDArray renderer as follows:
![](img/custom_debug_view.png)

Please make sure to:
- Check the "On-demand" option, which causes IntelliJ to only render the NDArray when you click on the variable.
- Change the "Use following expression" field to something like [toDebugString(100, 10, 10, 20)](https://javadoc.djl.ai/mxnet-engine/0.2.0/ai/djl/mxnet/engine/MxNDArray.html#toDebugString-int-int-int-int-)
if you want to adjust the range of NDArray's debug output.

## Common Problems

Please follow the [Troubleshooting](troubleshooting.md) guide for common problems and their solutions.
