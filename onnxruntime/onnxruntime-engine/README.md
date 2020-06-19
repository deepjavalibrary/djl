# DJL - ONNX Runtime engine implementation

## Overview
This module contains the ONNX Runtime implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the ONNX Runtime and make switching between frameworks difficult.
Even so, developers are not restricted from using engine-specific features.
For more information, see [NDManager#invoke()](https://javadoc.io/static/ai.djl/api/0.5.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

ONNX Runtime is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine for ONNX Runtime](../../docs/onnxruntime/hybrid_engine.md).

## Documentation

The latest javadocs can be found on the [djl.ai website](https://javadoc.io/doc/ai.djl.pytorch/pytorch-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
./gradlew javadoc
```
The javadocs output is built in the `build/doc/javadoc` folder.

## Installation
To use the Snapshot successfully, you need to add the snapshot repository:

```xml
<repositories>
    <repository>
        <id>djl.ai</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
    </repository>
</repositories>
```

for gradle:

```
repositories {
    maven {
        url "https://oss.sonatype.org/content/repositories/snapshots/"
    }
}
```

You can pull the ONNX Runtime engine from the central Maven repository by including the following dependency:

- ai.djl.onnxruntime:onnxruntime-engine:0.6.0-SNAPSHOT

```xml
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.6.0-SNAPSHOT</version>
    <scope>runtime</scope>
</dependency>
```
Besides the `onnxruntime-engine` library, you may also need to include the ONNX Runtime native library in your project.
All current provided ONNX Runtime native libraries were built from source based on the tag version.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the jars the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

- ai.djl.onnxruntime:onnxruntime-native-auto:1.3.0-SNAPSHOT

```xml
<dependency>
  <groupId>ai.djl.onnxruntime</groupId>
  <artifactId>onnxruntime-native-auto</artifactId>
  <version>1.3.0-SNAPSHOT</version>
  <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.onnxruntime:onnxruntime-native-cpu:1.3.0-SNAPSHOT:osx-x86_64

```xml
<dependency>
  <groupId>ai.djl.onnxruntime</groupId>
  <artifactId>onnxruntime-native-cpu</artifactId>
  <classifier>osx-x86_64</classifier>
  <version>1.3.0-SNAPSHOT</version>
  <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can use the following library:

- ai.djl.onnxruntime:onnxruntime-native-cpu:1.3.0-SNAPSHOT:linux-x86_64

```xml
<dependency>
  <groupId>ai.djl.onnxruntime</groupId>
  <artifactId>onnxruntime-native-cpu</artifactId>
  <classifier>linux-x86_64</classifier>
  <scope>runtime</scope>
  <version>1.3.0-SNAPSHOT</version>
</dependency>
```

### Windows

For the Windows platform, you can use the following library:

- ai.djl.onnxruntime:onnxruntime-native-cpu:1.3.0-SNAPSHOT:win-x86_64

```xml
<dependency>
  <groupId>ai.djl.onnxruntime</groupId>
  <artifactId>onnxruntime-native-cpu</artifactId>
  <classifier>win-x86_64</classifier>
  <scope>runtime</scope>
  <version>1.3.0-SNAPSHOT</version>
</dependency>
```
