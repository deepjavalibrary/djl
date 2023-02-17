# DJL - Apache MXNet engine implementation

## Overview

This module contains the Deep Java Library (DJL) EngineProvider for Apache MXNet.

We don't recommend that developers use classes in this module directly. Use of these classes
will couple your code with Apache MXNet and make switching between engines difficult. Even so,
developers are not restricted from using engine-specific features. For more information,
see [NDManager#invoke()](https://javadoc.io/static/ai.djl/api/0.21.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDArray:A-ai.djl.ndarray.NDArray:A-ai.djl.util.PairList-).

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.mxnet/mxnet-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.


## Installation

You can pull the MXNet engine from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-engine</artifactId>
    <version>0.21.0</version>
    <scope>runtime</scope>
</dependency>
```

By default, DJL will download the Apache MXNet native libraries into [cache folder](../../../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

If you do not want to rely on the download because you don't have network access at runtime or for other reasons, there are additional options.
The easiest option is to add a DJL native library package to your project dependencies.
The available packages for your platform can be found below.
Finally, you can also specify the path to a valid MXNet build using the `MXNET_LIBRARY_PATH` environment variable.

### macOS

For macOS, you can use the following library:

- ai.djl.mxnet:mxnet-native-mkl:1.9.1:osx-x86_64

    This package takes advantage of the Intel MKL library to boost performance.

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-native-mkl</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>1.9.1</version>
    <scope>runtime</scope>
</dependency>
```

### Linux

For the Linux platform, you can choose between CPU, GPU. If you have Nvidia [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

**Important:** Since cuda 11.0, you must install CUDNN and NCCL matches the CUDA version manually.
Apache MXNet no longer statically link CUDNN and NCCL in it.

Apache MXNet 1.9.1 cu112 package supports SM 5.0, 6.0, 7.0 and 8.0 cuda architectures.
Apache MXNet 1.9.1 cu102 package supports SM 3.0, 5.0, 6.0, 7.0 and 7.5 cuda architectures.

- ai.djl.mxnet:mxnet-native-cu112mkl:1.9.1:linux-x86_64 - CUDA 11.2
- ai.djl.mxnet:mxnet-native-cu102mkl:1.9.1:linux-x86_64 - CUDA 10.2

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-native-cu112mkl</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>1.9.1</version>
    <scope>runtime</scope>
</dependency>
```

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-native-cu102mkl</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>1.9.1</version>
    <scope>runtime</scope>
</dependency>
```

#### Linux CPU

- ai.djl.mxnet:mxnet-native-mkl:1.9.1:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-native-mkl</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>1.9.1</version>
</dependency>
```

### Windows

Apache MXNet requires Visual C++ Redistributable Packages. If you encounter an UnsatisfiedLinkError while using
DJL on Windows, please download and install
[Visual C++ 2019 Redistributable Packages](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and reboot.

For the windows platform, we support both CPU and GPU.
The CPU can be found using either the automatic runtime detection or through adding the CPU jar to your dependencies.
However, due to the size of the windows GPU native library, we do not offer GPU support through a dependency jar.
You can still access GPU on windows by using the [automatic runtime download](#installation).

#### Windows GPU

This package supports CUDA 11.2 and CUDA 10.2 for Windows.

### Windows CPU

- ai.djl.mxnet:mxnet-native-mkl:1.9.1:win-x86_64

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-native-mkl</artifactId>
    <classifier>win-x86_64</classifier>
    <scope>runtime</scope>
    <version>1.9.1</version>
</dependency>
```
