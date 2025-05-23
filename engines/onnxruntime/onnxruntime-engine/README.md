# DJL - ONNX Runtime engine implementation

## Overview
This module contains the Deep Java Library (DJL) EngineProvider for ONNX Runtime.

It is based off the [ONNX Runtime Deep Learning Framework](https://onnxruntime.ai/).


We don't recommend developers use classes within this module directly.
Use of these classes will couple your code to the ONNX Runtime and make switching between engines difficult.

ONNX Runtime is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.
For more information, see [Hybrid Engine](../../../docs/hybrid_engine.md).

## Documentation

The latest javadocs can be found on [here](https://javadoc.io/doc/ai.djl.onnxruntime/onnxruntime-engine/latest/index.html).

You can also build the latest javadocs locally using the following command:

```sh
# for Linux/macOS:
./gradlew javadoc

# for Windows:
..\..\gradlew javadoc
```
The javadocs output is generated in the `build/doc/javadoc` folder.

#### System Requirements

Read the [System Requirements](https://github.com/microsoft/onnxruntime/blob/master/README.md#system-requirements)
for the official ONNX Runtime project.

## Installation

> [!CAUTION]
> OnnxRuntime 1.21.0+ compiled with VC 14.42 for Windows, which has conflict with some of JDK distribution
> and will cause `UnsatisfiedLinkError`.
> We tested with `temurin17` and `zulu17`, they works on Windows:
> ```
> choco install temurin17
> choco install zulu17
> ```

You can pull the ONNX Runtime engine from the central Maven repository by including the following dependency:

- ai.djl.onnxruntime:onnxruntime-engine:0.33.0

```xml
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.33.0</version>
    <scope>runtime</scope>
</dependency>
```

This package by default depends on [com.microsoft.onnxruntime:onnxruntime](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)

### Install GPU package
If you want to use GPU, you can manually exclude com.microsoft.onnxruntime:onnxruntime
and add [com.microsoft.onnxruntime:onnxruntime_gpu](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)  
to your project.

Maven:

```xml
<dependency>
    <groupId>ai.djl.onnxruntime</groupId>
    <artifactId>onnxruntime-engine</artifactId>
    <version>0.33.0</version>
    <scope>runtime</scope>
    <exclusions>
        <exclusion>
            <groupId>com.microsoft.onnxruntime</groupId>
            <artifactId>onnxruntime</artifactId>
        </exclusion>
    </exclusions>
</dependency>
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime_gpu</artifactId>
    <version>1.21.1</version>
    <scope>runtime</scope>
</dependency>
```

Gradle:

```groovy
implementation("ai.djl.onnxruntime:onnxruntime-engine:0.33.0") {
    exclude group: "com.microsoft.onnxruntime", module: "onnxruntime"
}
implementation "com.microsoft.onnxruntime:onnxruntime_gpu:1.21.1"
```

#### Enable TensorRT execution

ONNXRuntime offers TensorRT execution as the backend. In DJL, user can specify the followings in the Criteria to enable:

```
optOption("ortDevice", "TensorRT")
```

The full example looks like the following:

```
String text = "What is deep learning?";
String url = "djl://ai.djl.huggingface.onnxruntime/TaylorAI/bge-micro-v2";
Criteria<String, float[]> criteria =
        Criteria.builder().setTypes(String.class, float[].class)
                .optModelUrls(url)
                .optOption("ortDevice", "TensorRT")
                .build();

try (ZooModel<String, float[]> model = criteria.loadModel();
        Predictor<String, float[]> predictor = model.newPredictor()) {
    float[] res = predictor.predict(text);
    Assert.assertEquals(res.length, 384);
}
```
