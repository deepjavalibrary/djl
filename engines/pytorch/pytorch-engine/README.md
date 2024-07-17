# DJL - PyTorch engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for PyTorch.

We don't recommend that developers use classes in this module directly.
Use of these classes will couple your code with PyTorch and make switching between frameworks difficult.

## Documentation

The latest javadocs can be found [here](https://javadoc.io/doc/ai.djl.pytorch/pytorch-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation
You can pull the PyTorch engine from the central Maven repository by including the following dependency:

- ai.djl.pytorch:pytorch-engine:0.29.0

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-engine</artifactId>
    <version>0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

Since DJL 0.14.0, `pytorch-engine` can load older version of pytorch native library. There are two
ways to specify PyTorch version:

1. Explicitly specify `pytorch-native-xxx` package version to override the version in the [BOM](../../../bom/README.md).
2. Sets environment variable: `PYTORCH_VERSION` to override the default package version.

### Supported PyTorch versions
The following table illustrates which pytorch version that DJL supports:

| PyTorch engine version | PyTorch native library version            |
|------------------------|-------------------------------------------|
| pytorch-engine:0.29.0  | 1.13.1, 2.1.2, 2.2.2, **2.3.1**           |
| pytorch-engine:0.28.0  | 1.13.1, 2.1.2, **2.2.2**                  |
| pytorch-engine:0.27.0  | 1.13.1, **2.1.1**                         |
| pytorch-engine:0.26.0  | 1.13.1, 2.0.1, **2.1.1**                  |
| pytorch-engine:0.25.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.24.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.23.0  | 1.11.0, 1.12.1, **1.13.1**, 2.0.1         |
| pytorch-engine:0.22.1  | 1.11.0, 1.12.1, **1.13.1**, 2.0.0         |
| pytorch-engine:0.21.0  | 1.11.0, 1.12.1, **1.13.1**                |
| pytorch-engine:0.20.0  | 1.11.0, 1.12.1, **1.13.0**                |
| pytorch-engine:0.19.0  | 1.10.0, 1.11.0, **1.12.1**                |
| pytorch-engine:0.18.0  | 1.9.1, 1.10.0, **1.11.0**                 |
| pytorch-engine:0.17.0  | 1.9.1, 1.10.0, 1.11.0                     |
| pytorch-engine:0.16.0  | 1.8.1, 1.9.1, 1.10.0                      |
| pytorch-engine:0.15.0  | pytorch-native-auto: 1.8.1, 1.9.1, 1.10.0 |
| pytorch-engine:0.14.0  | pytorch-native-auto: 1.8.1, 1.9.0, 1.9.1  |
| pytorch-engine:0.13.0  | pytorch-native-auto:1.9.0                 |
| pytorch-engine:0.12.0  | pytorch-native-auto:1.8.1                 |
| pytorch-engine:0.11.0  | pytorch-native-auto:1.8.1                 |
| pytorch-engine:0.10.0  | pytorch-native-auto:1.7.1                 |
| pytorch-engine:0.9.0   | pytorch-native-auto:1.7.0                 |
| pytorch-engine:0.8.0   | pytorch-native-auto:1.6.0                 |
| pytorch-engine:0.7.0   | pytorch-native-auto:1.6.0                 |
| pytorch-engine:0.6.0   | pytorch-native-auto:1.5.0                 |
| pytorch-engine:0.5.0   | pytorch-native-auto:1.4.0                 |
| pytorch-engine:0.4.0   | pytorch-native-auto:1.4.0                 |

### BOM support
We strongly recommend you to use [Bill of Materials (BOM)](../../../bom/README.md) to manage your dependencies.

By default, DJL will download the PyTorch native libraries into [cache folder](../../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

### CentOS 7 or Amazon Linux 2 support 
If you are running on an older operating system (like CentOS 7), you have to use
[precxx11 build](#for-pre-cxx11-build) or set system property to auto select for precxx11 binary:

```java
System.setProperty("PYTORCH_PRECXX11", "true");
```

or use System env

```shell
export PYTORCH_PRECXX11=true
```

If you don't have network access, you can add a offline native library package based on your platform
to avoid downloading the native libraries at runtime.

### Load your own PyTorch native library

If you installed PyTorch with python pip wheel, and you want to use your installed PyTorch,
you can set `PYTORCH_LIBRARY_PATH` environment variable, DJL will load your PyTorch native
library for the location you pointed to. You might also need set `PYTORCH_VERSION` and
`PYTORCH_FLAVOR` environment variable so DJL will use matching JNI for your PyTorch.

```shell
export PYTORCH_LIBRARY_PATH=/usr/lib/python3.10/site-packages/torch/lib

# Use latest PyTorch version that engine supported if PYTORCH_VERSION not set
export PYTORCH_VERSION=1.XX.X

# Use cpu-precxx11 if PYTORCH_FLAVOR not set
export PYTORCH_FLAVOR=cpu
```

### macOS
For macOS, you can use the following library:

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cpu:2.3.1:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

**Note:** PyTorch 1.13+ doesn't support mac 11 any more, you must use DJL 0.19.0 ane lower version.

### macOS M1
For macOS M1, you can use the following library:

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cpu:2.3.1:osx-aarch64

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>osx-aarch64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cu121:2.3.1:linux-x86_64 - CUDA 12.1

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cu121</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### Linux CPU

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cpu:2.3.1:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.3.1</version>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### For aarch64 build

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cpu-precxx11:2.3.1:linux-aarch64

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu-precxx11</artifactId>
    <classifier>linux-aarch64</classifier>
    <scope>runtime</scope>
    <version>2.3.1</version>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### For Pre-CXX11 build

We also provide packages for the system like CentOS 7/Ubuntu 14.04 with GLIBC >= 2.17.
All the package were built with GCC 7, we provided a newer `libstdc++.so.6.24` in the package that contains `CXXABI_1.3.9` to use the package successfully.

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cu121-precxx11:2.3.1:linux-x86_64 - CUDA 12.1
- ai.djl.pytorch:pytorch-native-cpu-precxx11:2.3.1:linux-x86_64   - CPU

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cu121-precxx11</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu-precxx11</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### Windows

PyTorch requires Visual C++ Redistributable Packages. If you encounter an UnsatisfiedLinkError while using
DJL on Windows, please download and install
[Visual C++ 2019 Redistributable Packages](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and reboot.

For the Windows platform, you can choose between CPU and GPU.

#### Windows GPU

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cu121:2.3.1:win-x86_64 - CUDA 12.1

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cu121</artifactId>
    <classifier>win-x86_64</classifier>
    <version>2.3.1</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```

### Windows CPU

- ai.djl.pytorch:pytorch-jni:2.3.1-0.29.0
- ai.djl.pytorch:pytorch-native-cpu:2.3.1:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-native-cpu</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.3.1</version>
</dependency>
<dependency>
    <groupId>ai.djl.pytorch</groupId>
    <artifactId>pytorch-jni</artifactId>
    <version>2.3.1-0.29.0</version>
    <scope>runtime</scope>
</dependency>
```
