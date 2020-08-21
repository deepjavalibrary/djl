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

## CPU Build

Use the following task to build PyTorch JNI library:

### Mac/Linux

```
./gradlew compileJNI
```

### Windows

```
gradlew compileJNI
```

This task will send a Jni library copy to `pytorch-engine` model to test locally.

## GPU build
Note: PyTorch C++ library requires CUDA path set in the system.

Use the following task to build pytorch JNI library for GPU:

### Mac/Linux

```
./gradlew compileJNIGPU
```

## Windows

```
gradlew compileJNIGPU
```

The task will build CUDA 10.1 by default, you can change the flavor in `compileJNIGPU` to `cu92` to use CUDA 9.2.

```
downloadBuild("win", "cu92")
```

### Format C++ code
It uses clang-format to format the code.

```
./gradlew formatCpp
```
