# DJL TensorFlow Engine

The DJL TensorFlow Engine allows you to run prediction with TensorFlow or Keras models using Java.
It has the following 4 modules:

1. [TensorFlow core api](https://github.com/awslabs/djl/tree/master/tensorflow/tensorflow-api): the TensorFlow 2.x java binding.
2. [TensorFlow engine](https://github.com/awslabs/djl/tree/master/tensorflow/tensorflow-engine): TensorFlow engine adapter for DJL high level APIs. (NDArray, Model, Predictor, etc)
3. [TensorFlow model zoo](https://github.com/awslabs/djl/tree/master/tensorflow/tensorflow-model-zoo): Includes pre-trained TensorFlow models and built-int translators for direct import and use.
4. [TensorFlow native auto](https://github.com/awslabs/djl/tree/master/tensorflow/tensorflow-native-auto): A placeholder to automatically detect your platform and download the correct native TensorFlow libraries for you.

## Installation
All TensorFlow modules are currently experimental and are only available in SNAPSHOTs.

Note:
1. The TensorFlow native auto module only supports detecting Mac OSX, Linux CPU and Linux GPU with CUDA version from 9.2 to 10.2.
2. For GPU usage, you need to install [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and  [cuDNN Library](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html).
3. You can use `-Dai.djl.default_engine=TensorFlow` to switch between different Engines DJL support.

### Gradle
For gradle usage, include the snapshot repository and add the 4 modules in your dependencies:
```
repositories {
    maven {
        url 'https://oss.sonatype.org/content/repositories/snapshots/'
    }
}

dependencies {
    implementation "ai.djl:api:0.5.0-SNAPSHOT"
    implementation "ai.djl.tensorflow:tensorflow-api:0.5.0-SNAPSHOT"
    implementation "ai.djl.tensorflow:tensorflow-engine:0.5.0-SNAPSHOT"
    implementation "ai.djl.tensorflow:tensorflow-model-zoo:0.5.0-SNAPSHOT"
    implementation "ai.djl.tensorflow:tensorflow-native-auto:2.1.0-a-SNAPSHOT"
}
```

### Maven

Same as gradle, add snapshot repository and include the 4 modules:
```xml
<repositories>
    <repository>
        <id>djl-snapshots</id>
        <url>https://oss.sonatype.org/content/repositories/snapshots/</url>
        <snapshots>
            <enabled>true</enabled>
        </snapshots>
    </repository>
</repositories>

<dependencies>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-api</artifactId>
      <version>0.5.0-SNAPSHOT</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-engine</artifactId>
      <version>0.5.0-SNAPSHOT</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-model-zoo</artifactId>
      <version>0.5.0-SNAPSHOT</version>
    </dependency>
    <dependency>
      <groupId>ai.djl.tensorflow</groupId>
      <artifactId>tensorflow-native-auto</artifactId>
      <version>2.1.0-a-SNAPSHOT</version>
    </dependency>
</dependencies>
```

# Documentation

For more TensorFlow Engine documentation please refer to our [docs folder](../docs/tensorflow)


