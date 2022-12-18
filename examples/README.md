# Examples

This module contains examples to demonstrate use of the Deep Java Library (DJL).
You can find more examples from our [djl-demo github repo](https://github.com/deepjavalibrary/djl-demo).

The following examples are included for training:

- [Train your first model](docs/train_mnist_mlp.md)
- [Transfer learning on cifar10](docs/train_cifar10_resnet.md)
- [Transfer learning on freshfruit](docs/train_transfer_fresh_fruit.md)
- [Train SSD model example](docs/train_pikachu_ssd.md)
- [Multi-label dataset training example](docs/train_captcha.md)


The following examples are included for inference:

- [Image classification example](docs/image_classification.md)
- [Single-shot object detection example](docs/object_detection.md)
- [Face detection example](docs/face_detection.md)
- [Face recognition example](docs/face_recognition.md)
- [Instance segmentation example](docs/instance_segmentation.md)
- [Semantic segmentation example](docs/semantic_segmentation.md)
- [Pose estimation example](docs/pose_estimation.md)
- [Action recognition example](docs/action_recognition.md)
- [Image generation with BigGAN example](docs/biggan.md)
- [Image enhancement with Super-Resolution example](docs/super_resolution.md)
- [Bert question and answer example](docs/BERT_question_and_answer.md)
- [Speech recognition example](docs/speech_recognition.md)

These examples focus on the overall experience of training and inference. We keep components
that are reusable within separate modules for other users to take advantage of in their own
applications. For examples and references on creating datasets, look at the
[basic dataset module](https://github.com/deepjavalibrary/djl/tree/master/basicdataset).
For examples and references on building models and translators, look in our
[basic model zoo](https://github.com/deepjavalibrary/djl/tree/master/model-zoo).

You may be able to find more translator examples in our engine specific model zoos:
[Apache MXNet](https://github.com/deepjavalibrary/djl/tree/master/engines/mxnet/mxnet-model-zoo),
[PyTorch](https://github.com/deepjavalibrary/djl/tree/master/engines/pytorch/pytorch-model-zoo),
and [TensorFlow](https://github.com/deepjavalibrary/djl/tree/master/engines/tensorflow/tensorflow-model-zoo).

More examples and demos of applications featuring DJL are located in our [demo repository](https://github.com/deepjavalibrary/djl-demo).

## Prerequisites

* You need to have Java Development Kit version 8 or later installed on your system. For more information, see [Setup](../docs/development/setup.md).
* You should be familiar with the API documentation in the DJL [Javadoc](https://javadoc.io/doc/ai.djl/api/latest/index.html).


# Getting started: 30 seconds to run an example

## Building with the command line

This example supports building with both Gradle and Maven. To build, use either of the following commands:

* Gradle build

```sh
cd examples

# for Linux/macOS:
./gradlew jar

# for Windows:
..\gradlew jar
```

* Maven build

```sh
cd examples
mvn package -DskipTests
```

### Run example code
With the gradle `application` plugin you can execute example code directly.
For more information on running each example, see the example's documentation.

The following command executes an object detection example:

* Gradle

```sh
cd examples

# for Linux/macOS:
./gradlew run

# for Windows:
..\gradlew run
```

* Maven

```sh
cd examples
mvn clean package -DskipTests
mvn exec:java
```

## Engine selection

DJL is engine agnostic, so it's capable of supporting different backends. DJL by default will select
proper native library for you automatically and download those libraries from internet. If your
production environment doesn't have network access, you can distribute DJL's offline native packages
together with your application to avoid download engine native libraries at runtime.

With Apache MXNet, PyTorch, TensorFlow and ONNX Runtime, you can choose different
builds of the native library.
You can also see:

- [MXNet Engine](../engines/mxnet/mxnet-engine/README.md)
- [PyTorch Engine](../engines/pytorch/pytorch-engine/README.md)
- [TensorFlow Engine](../engines/tensorflow/tensorflow-engine/README.md)
- [ONNX Runtime Engine](../engines/onnxruntime/onnxruntime-engine/README.md)
