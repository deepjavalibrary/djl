# Hybrid Engine for ONNX Runtime

## Introduction

ONNX Runtime is a DL library with limited support for NDArray operations.
Currently, it only covers the basic NDArray creation methods. To better support the necessary preprocessing and postprocessing,
you can use one of the other Engine along with it to run in a hybrid mode.

The minimum dependencies for the ONNX Runtime Engine are:

#### gradle
```
runtimeOnly "ai.djl.onnxruntime:onnxruntime-engine:0.6.0-SNAPSHOT"
runtimeOnly "ai.djl.onnxruntime:onnxruntime-native-auto:1.3.0-SNAPSHOT"
```

To use it along with MXNet for additional API support, add the following two dependencies:
```
runtimeOnly "ai.djl.mxnet:mxnet-engine:0.6.0-SNAPSHOT"
runtimeOnly "ai.djl.mxnet:mxnet-native-auto:1.7.0-a"
```
You can also use PyTorch or TensorFlow Engine as the supplemental engine by adding their corresponding dependencies.

## How Hybrid works

Internally, DJL will find two or more engines available. When you start using the ONNX Runtime Engine,
the engine will search for additional Engines. If it finds any, all NDArrays will be created in
the supplemental engine. After that, the NDArrays will only be converted to ONNX Runtime NDArrays right before inference.

If a second Engine is not found, the ONNX Runtime Engine will use its own NDManager and NDArray class to support
limited NDArray creation methods.
