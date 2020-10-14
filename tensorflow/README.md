# DJL TensorFlow Engine

This directory contains the Deep Java Library (DJL) EngineProvider for TensorFlow.

It is based off the [TensorFlow Deep Learning Framework](https://www.tensorflow.org/).

The DJL TensorFlow Engine allows you to run prediction with TensorFlow or Keras models using Java.
It has the following 4 modules:

1. [TensorFlow core api](tensorflow-api/README.md): the TensorFlow 2.x java binding.
2. [TensorFlow engine](tensorflow-engine/README.md): TensorFlow engine adapter for DJL high level APIs. (NDArray, Model, Predictor, etc)
3. [TensorFlow model zoo](tensorflow-model-zoo/README.md): Includes pre-trained TensorFlow models and built-int translators for direct import and use.
4. [TensorFlow native auto](tensorflow-native/README.md): A placeholder to automatically detect your platform and download the correct native TensorFlow libraries for you.

Refer to [How to import TensorFlow models](https://docs.djl.ai/docs/tensorflow/how_to_import_tensorflow_models_in_DJL.html) for loading TF models in DJL.

## Installation
You can pull the TensorFlow engine from the central Maven repository by including the following dependency:

- ai.djl.tensorflow:tensorflow-engine:0.8.0

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-engine</artifactId>
    <version>0.8.0</version>
    <scope>runtime</scope>
</dependency>
```
Besides the `tensorflow-engine` library, you may also need to include the TensorFlow native library in your project.

Choose a native library based on your platform and needs:

### Automatic (Recommended)

We offer an automatic option that will download the native libraries into [cache folder](../docs/development/cache_management.md) the first time you run DJL.
It will automatically determine the appropriate jars for your system based on the platform and GPU support.

- ai.djl.tensorflow:tensorflow-native-auto:2.3.0

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-auto</artifactId>
    <version>2.3.0</version>
    <scope>runtime</scope>
</dependency>
```

### macOS
For macOS, you can use the following library:

- ai.djl.tensorflow:tensorflow-native-cpu:2.3.0:osx-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>osx-x86_64</classifier>
    <version>2.3.0</version>
    <scope>runtime</scope>
</dependency>
```

### Linux
For the Linux platform, you can choose between CPU, GPU. If you have NVIDIA [CUDA](https://en.wikipedia.org/wiki/CUDA)
installed on your GPU machine, you can use one of the following library:

#### Linux GPU

- ai.djl.tensorflow:tensorflow-native-cu101:2.3.0:linux-x86_64 - CUDA 10.1

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cu101</artifactId>
    <classifier>linux-x86_64</classifier>
    <version>2.3.0</version>
    <scope>runtime</scope>
</dependency>
```

*Note: If you have gcc version less than 7.0, you will need to do either one of the following because
TensorFlow requires a higher version of `libstdc++`:

* Set `LD_LIBRARY_PATH` environment variable to the TensorFlow native library path
where we included a higher version of `libstdc++` for you.

```bash
export LD_LIBRARY_PATH=$HOME/.tensorflow/cache/2.3.0-cu101-linux-x86_64/:$LD_LIBRARY_PATH
```

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

### Linux CPU

- ai.djl.tensorflow:tensorflow-native-cpu:2.3.0:linux-x86_64

```xml
<dependency>
    <groupId>ai.djl.tensorflow</groupId>
    <artifactId>tensorflow-native-cpu</artifactId>
    <classifier>linux-x86_64</classifier>
    <scope>runtime</scope>
    <version>2.3.0</version>
</dependency>
```
