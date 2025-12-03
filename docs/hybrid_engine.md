# Hybrid Engine

## Introduction

Many of DJL engines only has limited support for NDArray operations. Here is a list of such engines:

- [ONNX Runtime](../engines/onnxruntime/onnxruntime-engine/README.md)
- [XGBoost](../engines/ml/xgboost/README.md)
- [LightGBM](../engines/ml/lightgbm/README.md)

Currently, those engines only covers the basic NDArray creation methods. To better support the
necessary preprocessing and postprocessing, you can use one of the full Engines along with it
to run in a hybrid mode:

- [MXNet](../engines/mxnet/README.md) - full NDArray support
- [PyTorch](../engines/pytorch/README.md) - support most of NDArray operations
- [TensorFlow](../engines/tensorflow/README.md) - support most of NDArray operations


To use it along with Apache MXNet for additional API support, add the following two dependencies:

```
runtimeOnly "ai.djl.mxnet:mxnet-engine:0.35.1"
```

You can also use PyTorch or TensorFlow Engine as the supplemental engine by adding their corresponding dependencies.

```
runtimeOnly "ai.djl.pytorch:pytorch-engine:0.35.1"
```

```
runtimeOnly "ai.djl.tensorflow:tensorflow-engine:0.35.1"
```

## How Hybrid works

Internally, DJL will find two or more engines available. When you start using the hybrid engine,
DJL will search for additional full Engines. Whenever an unsupported NDArray operation is invoked,
it will delegate to alternative full engine to run the operation.
If you don't need preprocessing/postprocessing and trying to avoid loading another engine
at runtime, you can disable this behavior by setting the following system properties:

```
# disable hybrid engine for OnnxRuntime
System.setProperty("ai.djl.onnx.disable_alternative", "true");
```


