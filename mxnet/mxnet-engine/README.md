# DJL - MXNet engine implementation

## Overview

This module contains the MXNet implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend that developers use classes in this module directly. Use of these classes will couple your code with MXNet and make switching between frameworks difficult. Even so, developers are not restricted from using engine-specific features. For more information, see [NDManager#invoke()](https://djl-ai.s3.amazonaws.com/java-api/0.2.0/api/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

## Installation
You can pull the MXNet engine from the central Maven repository by including the following dependency:

```xml
<dependency>
    <groupId>ai.djl.mxnet</groupId>
    <artifactId>mxnet-engine</artifactId>
    <version>0.2.0</version>
    <scope>runtime</scope>
</dependency>
```

Besides the `mxnet-engine` library, you may also need to include the MXNet native library in your project.
Choose a native library based on your platform and needs:

### macOS
For macOS, you can choose between the following two libraries:

- org.apache.mxnet:mxnet-native-mkl:1.6.0:osx-x86_64

    This package takes advantage of the Intel MKL library to boost performance.
```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-mkl</artifactId>
      <classifier>osx-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

- org.apache.mxnet:mxnet-native:1.6.0:osx-x86_64

    This package is a compact version of MXNet that minimizes your package size.
```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native</artifactId>
      <classifier>osx-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

### Linux
For the Linux platform, you can choose between CPU, MKL, CUDA and CUDA+MKL combinations:

- org.apache.mxnet:mxnet-native-cu101mkl:1.6.0:linux-x86_64
- org.apache.mxnet:mxnet-native-cu92mkl:1.6.0:linux-x86_64
- org.apache.mxnet:mxnet-native-cu90mkl:1.6.0:linux-x86_64

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu101mkl</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu92mkl</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu90mkl</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

- org.apache.mxnet:mxnet-native-cu101:1.6.0:linux-x86_64
- org.apache.mxnet:mxnet-native-cu92:1.6.0:linux-x86_64
- org.apache.mxnet:mxnet-native-cu90:1.6.0:linux-x86_64

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu101</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu92</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-cu90</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```

- org.apache.mxnet:mxnet-native-mkl:1.6.0:linux-x86_64
- org.apache.mxnet:mxnet-native:1.6.0:linux-x86_64

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native-mkl</artifactId>
      <classifier>linux-x86_64</classifier>
      <scope>runtime</scope>
      <version>1.6.0</version>
    </dependency>
```

```xml
    <dependency>
      <groupId>org.apache.mxnet</groupId>
      <artifactId>mxnet-native</artifactId>
      <classifier>linux-x86_64</classifier>
      <version>1.6.0</version>
      <scope>runtime</scope>
    </dependency>
```


### Windows

Coming soon

