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

### Note
If you are using TensorFlow 2.3.0 engine on GPU and you have gcc version less than 7.0, you will need to do either one of the following because TensorFlow requires a higher version of `libstdc++`:
* Set `LD_LIBRARY_PATH` environment variable to the TensorFlow native library path
where we included a higher version of `libstdc++` for you.

For GPU with CUDA 10.1:
```bash
export LD_LIBRARY_PATH=$HOME/.tensorflow/cache/2.3.0-SNAPSHOT-cu101-linux-x86_64/:$LD_LIBRARY_PATH
```
For GPU with CUDA 10.2:
```bash
export LD_LIBRARY_PATH=$HOME/.tensorflow/cache/2.3.0-SNAPSHOT-cu102-linux-x86_64/:$LD_LIBRARY_PATH
```
OR
* upgrade your gcc to gcc7+, you can use the following commands:
```bash
sudo apt-get update && \
sudo apt-get install -y software-properties-common && \
sudo add-apt-repository ppa:ubuntu-toolchain-r/test && \
sudo apt-get update && \
sudo apt-get install -y gcc-7 g++-7 && \
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 60 && \
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60
```

 
### Gradle
For gradle usage, include the snapshot repository and add the 4 modules in your dependencies:

```
repositories {
    jcenter()
}

dependencies {
    implementation "ai.djl.tensorflow:tensorflow-engine:0.7.0"
    implementation "ai.djl.tensorflow:tensorflow-model-zoo:0.7.0"
    implementation "ai.djl.tensorflow:tensorflow-native-auto:2.3.0"
}
```

### Maven

Same as gradle, just include the 4 modules:

```xml
<dependencies>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-engine</artifactId>
      <version>0.7.0</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-model-zoo</artifactId>
      <version>0.7.0</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-native-auto</artifactId>
      <version>2.3.0</version>
    </dependency>
</dependencies>
```
