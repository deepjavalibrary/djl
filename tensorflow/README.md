# DJL TensorFlow Engine

This directory contains the Deep Java Library (DJL) EngineProvider for TensorFlow.

It is based off the [TensorFlow Deep Learning Framework](https://www.tensorflow.org/).

The DJL TensorFlow Engine allows you to run prediction with TensorFlow or Keras models using Java.
It has the following 4 modules:

1. [TensorFlow core api](tensorflow-api/README.md): the TensorFlow 2.x java binding.
2. [TensorFlow engine](tensorflow-engine/README.md): TensorFlow engine adapter for DJL high level APIs. (NDArray, Model, Predictor, etc)
3. [TensorFlow model zoo](tensorflow-model-zoo/README.md): Includes pre-trained TensorFlow models and built-int translators for direct import and use.
4. [TensorFlow native auto](tensorflow-native-auto/README.md): A placeholder to automatically detect your platform and download the correct native TensorFlow libraries for you.

## Installation
You can pull the TensorFlow engine from the central Maven repository.
Note:
1. The TensorFlow native auto module only supports detecting Mac OSX, Linux CPU and Linux GPU with CUDA version from 9.2 to 10.2.
2. For GPU usage, you need to install [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and  [cuDNN Library](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).
3. You can use `-Dai.djl.default_engine=TensorFlow` to switch between different Engines DJL support.

### Gradle
For gradle usage, include the snapshot repository and add the 4 modules in your dependencies:

```
repositories {
    jcenter()
}

dependencies {
    implementation "ai.djl.tensorflow:tensorflow-engine:0.6.0"
    implementation "ai.djl.tensorflow:tensorflow-model-zoo:0.6.0"
    implementation "ai.djl.tensorflow:tensorflow-native-auto:2.2.0"
}
```

### Maven

Same as gradle, just include the 4 modules:

```xml
<dependencies>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-engine</artifactId>
      <version>0.6.0</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-model-zoo</artifactId>
      <version>0.6.0</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-native-auto</artifactId>
      <version>2.2.0</version>
    </dependency>
</dependencies>
```
