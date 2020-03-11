# DJL - PyTorch native Library

## Introduction
This project builds the JNI layer for Java to call PyTorch C++ APIs.
You can find more information in the `src`.

## Prerequisite
You need to install `cmake` and C++ compiler on your machine in order to build

### Linux
```
apt install cmake g++
```

## Build

Use the following command to build pytorch JNI library:

### Mac/Linux
```
./gradlew compileJNI
```

### Windows
```
gradlew compileJNI
```
