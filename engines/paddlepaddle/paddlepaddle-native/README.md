# DJL - PaddlePaddle native library

This module publishes the PaddlePaddle native libraries to Maven central repository.

## Prerequisite
You need to install `cmake` and C++ compiler on your machine in order to build

### Linux

```sh
apt install cmake g++
```

### Windows

Visual Studio 2017/2019 with C++ development toolkit is required to compile.

## CPU Build

Use the following task to build PaddlePaddle JNI library:

### Mac/Linux

```sh
./gradlew compileJNI
```

### Windows

```cmd
gradlew compileJNI
```

This task will send a Jni library copy to `paddlepaddle-engine` model to test locally.
