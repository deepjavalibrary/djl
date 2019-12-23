# Development Guideline

## Introduction

Thank you for your interest in contributing to the Deep Java Library (DJL).
In this document, we will cover everything you need to build, test, and debug your code when developing DJL.

Many of us use the [IntelliJ IDEA IDE](https://www.jetbrains.com/idea/) to develop DJL and we will sometimes mention it. However, there is no requirement to use this IDE.

## Coding Conventions

When writing code for DJL, we usually try to follow standard Java coding conventions. In addition, here are some other conventions we use:

- For builders, use setXXX for required values and optXXX for optional ones
- Follow the example in `Convolution` and `Conv2D` when making extendable builders

Alongside these conventions, we have a number of checks that are run including PMD, SpotBugs, and Checkstyle. These can all be verified by running the gradle `build` target. Instructions for fixing any problems will be given by the relevant tool.

We also follow the [AOSP Java Code Style](https://source.android.com/setup/contribute/code-style). See [here](https://github.com/google/google-java-format) for plugins that can help setup your IDE to use this style. The formatting is checked very strictly. Failing the formatting check will look like:
```
> Task :api:verifyJava FAILED

FAILURE: Build failed with an exception.

* Where:
Script '/Volumes/Unix/projects/Joule/tools/gradle/formatter.gradle' line: 57

* What went wrong:
Execution failed for task ':api:verifyJava'.
> File not formatted: /Volumes/Unix/projects/Joule/api/src/main/java/ai/djl/nn/convolutional/Conv2D.java
```
If you do fail the format check, the easiest way to resolve it is to run the gradle `formatJava` target to reformat your code.

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

## Run examples in IntelliJ
Before you run any examples in IntelliJ, configure your Application template as follows:
1. Navigate to IntelliJ menu and select "Run". Select "Edit configurations...".
2. Expand "Template" on the left side list, and select "Application".
3. Change the "Working directory:" value to: "$MODULE_WORKING_DIR$".
4. Select "OK" to save the template.

Navigate to the 'examples' module. Open the class that you want to execute (e.g. ai.djl.examples.inference.ObjectDetection).
Select the triangle at the class declaration line. A popup menu appears with 3 items:
- Run 'ObjectDetection.main()'
- Debug 'ObjectDetection.main()'
- Run 'ObjectDetection.main()' with coverage

Select "Run 'ObjectDetection.main()'". IntelliJ executes the ObjectDetection example.

If you manually create a run configuration or the existing configuration failed to execute
due to a missing example resource, you can edit the configuration. Change the "Working directory:"
value to: $MODULE_WORKING_DIR$ to fix the issue.

## Debug

When debugging a DJL application in IntelliJ, it is often helpful to inspect your NDArray variables. Because NDArrays may contain
a large number of elements, rendering them can be resource-heavy and cause the debugger to hang.

IntelliJ allows you to [customize the data view in the Debug Tools](https://www.jetbrains.com/help/idea/customize-data-views.html).
You can create your own NDArray renderer as follows:
![](img/custom_debug_view.png)

Please make sure to:
- Check the "On-demand" option, which causes IntelliJ to only render the NDArray when you click on the variable.
- Change the "Use following expression" field to something like [toDebugString(100, 10, 10, 20)](https://javadoc.djl.ai/mxnet-engine/0.2.1/ai/djl/mxnet/engine/MxNDArray.html#toDebugString-int-int-int-int-)
if you want to adjust the range of NDArray's debug output.

## Common Problems

Please follow the [Troubleshooting](troubleshooting.md) guide for common problems and their solutions.
