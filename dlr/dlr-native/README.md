# DJL - DLR native Library

## Introduction
This project builds the JNI layer for Java to call DLR C APIs.
You can find more information in the `src`.

## Prerequisite
You need to install `cmake` and C++ compiler on your machine in order to build

### Linux

```
apt install cmake g++
```

## CPU Build

Use the following task to build DLR JNI library:

### Mac/Linux

```
./gradlew compileJNI
```

* Note that the windows is not supported.

This task will send a JNI library copy to `dlr-engine` model to test locally.

## GPU build

* GPU is not supported.

### Format C++ code
It uses clang-format to format the code.

```
./gradlew formatCpp
```
