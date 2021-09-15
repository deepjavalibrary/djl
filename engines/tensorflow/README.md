# DJL TensorFlow Engine

This directory contains the Deep Java Library (DJL) EngineProvider for TensorFlow.

It is based off the [TensorFlow Deep Learning Framework](https://www.tensorflow.org/).

The DJL TensorFlow Engine allows you to run prediction with TensorFlow or Keras models using Java.
Refer to [How to import TensorFlow models](https://docs.djl.ai/docs/tensorflow/how_to_import_tensorflow_models_in_DJL.html) for loading TF models in DJL.

## Modules

- [TensorFlow core api](tensorflow-api/README.md): the TensorFlow 2.x java binding.
- [TensorFlow engine](tensorflow-engine/README.md): TensorFlow engine adapter for DJL high level APIs. (NDArray, Model, Predictor, etc)
- [TensorFlow model zoo](tensorflow-model-zoo/README.md): Includes pre-trained TensorFlow models and built-int translators for direct import and use.
- [TensorFlow native library](tensorflow-native/README.md): A placeholder to automatically detect your platform and download the correct native TensorFlow libraries for you.
