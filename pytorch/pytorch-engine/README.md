# DJL - PyTorch engine implementation

## Overview
This module contains the PyTorch implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with PyTorch and make switching between frameworks difficult.
Even so, developers are not restricted from using engine-specific features.
For more information, see [NDManager#invoke()](https://javadoc.io/static/ai.djl/api/0.5.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.pytorch/pytorch-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation
You can pull the PyTorch engine from the central Maven repository by including the following dependency:

- ai.djl.pytorch:pytorch-engine:0.5.0

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-engine</artifactId>
    <version>0.5.0</version>
    <scope>runtime</scope>
</dependency>
```
Besides the `pytorch-engine` library, you may also need to include the PyTorch native library in your project.
All current provided PyTorch native libraries are downloaded from [PyTorch C++ distribution](https://pytorch.org/get-started/locally/#start-locally).

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the jars the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

- ai.djl.pytorch:pytorch-native-auto:1.5.0

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-auto</artifactId>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.pytorch:pytorch-native-cpu:1.5.0:osx-x86_64

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cpu</artifactId>
      <classifier>osx-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

GPU:
- ai.djl.pytorch:pytorch-native-cu102:1.5.0:linux-x86_64 - CUDA 10.2
- ai.djl.pytorch:pytorch-native-cu101:1.5.0:linux-x86_64 - CUDA 10.1
- ai.djl.pytorch:pytorch-native-cu92:1.5.0:linux-x86_64 - CUDA 9.2

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cu102</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cu101</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cu92</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

CPU
- ai.djl.pytorch:pytorch-native-cpu:1.5.0:linux-x86_64

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cpu</artifactId>
      <classifier>linux-x86_64</classifier>
      <scope>runtime</scope>
      <version>1.5.0</version>
    </dependency>
```

### For Pre-CXX11 build

We also provide packages for the system like CentOS 7 with GLIBC > 2.17.
All the package were built with GCC 7, user may need to upgrade the gcc version to 5 as a minimum
or provide a newer `libstdc++.so.6` that contains `CXXABI_1.3.9` to use the package successfully.

You can use the following line to check if your system `libstdc++.so.6` meets requirement:
```
strings /<path-to-the-file>/libstdc++.so.6 | grep CXXABI
```

User are required to use the corresponding `pytorch-engine` package along with the native package.

- ai.djl.pytorch:pytorch-engine-precxx11:0.6.0-SNAPSHOT

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-engine-precxx11</artifactId>
    <version>0.6.0-SNAPSHOT</version>
    <scope>runtime</scope>
</dependency>
```

For the native packages:

CPU
- ai.djl.pytorch:pytorch-native-cpu-precxx11:1.5.0:linux-x86_64

GPU:
- ai.djl.pytorch:pytorch-native-cu102-precxx11:1.5.0:linux-x86_64 - CUDA 10.2
- ai.djl.pytorch:pytorch-native-cu101-precxx11:1.5.0:linux-x86_64 - CUDA 10.1
- ai.djl.pytorch:pytorch-native-cu92-precxx11:1.5.0:linux-x86_64 - CUDA 9.2


### Windows

For the Windows platform, you can choose between CPU and GPU.

GPU:

- ai.djl.pytorch:pytorch-native-cu101:1.5.0:win-x86_64
- ai.djl.pytorch:pytorch-native-cu92:1.5.0:win-x86_64

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cu101</artifactId>
      <classifier>win-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cu92</artifactId>
      <classifier>win-x86_64</classifier>
      <version>1.5.0</version>
      <scope>runtime</scope>
    </dependency>
```

CPU
- ai.djl.pytorch:pytorch-native-cpu:1.5.0:win-x86_64

```xml
    <dependency>
      <groupId>ai.djl.pytorch</groupId>
      <artifactId>pytorch-native-cpu</artifactId>
      <classifier>win-x86_64</classifier>
      <scope>runtime</scope>
      <version>1.5.0</version>
    </dependency>
```
