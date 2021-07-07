# DJL dependency management

This document outlines how DJL manages its dependencies and how developers can find DJL packages they need.

## External dependencies

We try our best to minimize the external dependencies. The core DJL library only has 4 external dependencies:

- [sl4fj-api](https://mvnrepository.com/artifact/ai.djl/api)
- [Gson](https://mvnrepository.com/artifact/com.google.code.gson/gson)
- [JNA](https://mvnrepository.com/artifact/net.java.dev.jna/jna)
- [apache-commons-compress](https://mvnrepository.com/artifact/org.apache.commons/commons-compress)

## Which DJL package do I need?

Although DJL itself has many packages, but in a real production environment, the minimal dependencies can be:

- ai.djl:api
- one of the engine package (e.g. ai.djl.pytorch:pytorch-engine)
- native library of selected engine (e.g. ai.djl.pytorch:pytorch-native-auto)

We don't recommend include more than one engine into your project unless you need both of them. DJL will load
all engines into memory as long as they are in classpath, and those engines usually consume significant of memory.

## Using Bill of Materials module (BOM)

To automatically manage DJL packages' versions, we recommend you use the
[Bill of Materials](https://search.maven.org/search?q=g:ai.djl%20AND%20a:bom).
It can simplify your project's dependency management and make sure Maven picks the compatible versions of DJL modules.

See [How to use DJL's BOM](../../bom/README.md#how-to-use-djls-bom) for detail.

## List of DJL packages published on Maven Central

| Group ID | Artifact ID          | Description       |
|----------|----------------------|-------------------|
| [ai.djl](https://search.maven.org/search?q=g:ai.djl) | [api](../../api/README.md#installation) | DJL core API, contains top level, engine-agnostic classes for both inference and training |
| | [bom](../../bom/README.md) | Bill of Materials (BOM), used to manage DJL package dependencies |
| | [basicdataset](../../basicdataset/README.md#installation) | Contains a collection of built-in datasets |
| | [model-zoo](../../model-zoo/README.md#installation) | Contains a collection of built-in engine-agnostic models |
| | [serving](../../serving/serving/README.md#installation) | Contains an universal model server implementation |
| | examples (deprecated) | Contains DJL examples |
| | repository (deprecated, use api instead) | Contain classes for DJL Repository API. The package is moved into api in newer releases |
| | | |
| [ai.djl.mxnet](https://search.maven.org/search?q=g:ai.djl.mxnet) | [mxnet-engine](../../mxnet/mxnet-engine/README.md#installation) | Apache MXNet engine adapter |
| | [mxnet-model-zoo](../../mxnet/mxnet-model-zoo/README.md#installation) | Contains state of the art Apache MXNet symbolic models |
| | [mxnet-native-auto](../../mxnet/mxnet-engine/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [mxnet-native-mkl(osx-x86_64)](../../mxnet/mxnet-engine/README.md#macos) | Contains Apache MXNet native library for macOS |
| | [mxnet-native-mkl(win-x86_64)](../../mxnet/mxnet-engine/README.md#windows-cpu) | Contains Apache MXNet native library for Windows |
| | [mxnet-native-mkl(linux-x86_64)](../../mxnet/mxnet-engine/README.md#linux-cpu) | Contains Apache MXNet native library for Linux |
| | [mxnet-native-cu110(linux-x86_64)](../../mxnet/mxnet-engine/README.md#linux-gpu) | Contains Apache MXNet native library for Linux with CUDA 11.0|
| | [mxnet-native-cu102(linux-x86_64)](../../mxnet/mxnet-engine/README.md#linux-gpu) | Contains Apache MXNet native library for Linux with CUDA 10.2|
| | mxnet-native-cu101(linux-x86_64) (deprecated) | Contains Apache MXNet native library <= 1.7.0 for Linux with CUDA 10.1 |
| | mxnet-native-cu92(linux-x86_64) (deprecated) | Contains Apache MXNet native library <= 1.6.0 for Linux with CUDA 9.2 |
| | | |
| [ai.djl.pytorch](https://search.maven.org/search?q=g:ai.djl.pytorch) | [pytorch-engine](../../pytorch/pytorch-engine/README.md#installation) | PyTorch engine adapter |
| | [pytorch-model-zoo](../../pytorch/pytorch-model-zoo/README.md#installation) | Contains state of the art PyTorch torch script models |
| | [pytorch-native-auto](../../pytorch/pytorch-engine/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [pytorch-native-cpu(osx-x86_64)](../../pytorch/pytorch-engine/README.md#macos) | Contains PyTorch native library for macOS |
| | [pytorch-native-cpu(win-x86_64)](../../pytorch/pytorch-engine/README.md#windows-cpu) | Contains PyTorch native library for Windows |
| | [pytorch-native-cpu(linux-x86_64)](../../pytorch/pytorch-engine/README.md#linux-cpu) | Contains PyTorch native library for Linux |
| | [pytorch-native-cu111(linux-x86_64)](../../pytorch/pytorch-engine/README.md#linux-gpu) | Contains PyTorch native library for Linux with CUDA 11.1 |
| | [pytorch-native-cu111(win-x86_64)](../../pytorch/pytorch-engine/README.md#windows-gpu) | Contains PyTorch native library for Windows with CUDA 11.1 |
| | [pytorch-native-cu102(linux-x86_64)](../../pytorch/pytorch-engine/README.md#linux-gpu) | Contains PyTorch native library for Linux with CUDA 10.2 |
| | [pytorch-native-cu102(win-x86_64)](../../pytorch/pytorch-engine/README.md#windows-gpu) | Contains PyTorch native library for Windows with CUDA 10.2 |
| | [pytorch-native-cpu-precxx11 (linux-x86_64)](../../pytorch/pytorch-engine/README.md#for-pre-cxx11-build) | Contains PyTorch native library for centOS 7 and Ubuntu 14.04 |
| | pytorch-native-cu101(linux-x86_64) (deprecated) | Contains PyTorch native library <= 1.7.1 for Linux with CUDA 10.1 |
| | pytorch-native-cu101(win-x86_64) (deprecated) | Contains PyTorch native library <= 1.7.1 for Windows with CUDA 10.1 |
| | pytorch-native-cu92(linux-x86_64) (deprecated) | Contains PyTorch native library <= 1.6.0 for Linux with CUDA 9.2 |
| | pytorch-native-cu92(win-x86_64) (deprecated) | Contains PyTorch native library <= 1.6.0 for Windows with CUDA 9.2 |
| | pytorch-engine-precxx11 (deprecated) | PyTorch engine adapter <= 0.9.0 specific for precxx11 environment like CentOs 7 and Ubuntu 14.04 |
| | | |
| [ai.djl.tensorflow](https://search.maven.org/search?q=g:ai.djl.tensorflow) | [tensorflow-engine](../../tensorflow/tensorflow-engine/README.md#installation) | TensorFlow engine adapter |
| | [tensorflow-api](../../tensorflow/tensorflow-api/README.md#installation) | A redistribution of tensorFlow core java API 2.x |
| | [tensorflow-model-zoo](../../tensorflow/tensorflow-model-zoo/README.md#installation) | Contains state of the art TensorFlow saved bundle models |
| | [tensorflow-native-auto](../../tensorflow/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [tensorflow-native-cpu(osx-x86_64)](../../tensorflow/README.md#macos) | Contains TensorFlow native library for macOS |
| | [tensorflow-native-cpu(linux-x86_64)](../../tensorflow/README.md#linux-cpu) | Contains TensorFlow native library for Linux |
| | [tensorflow-native-cpu(win-x86_64)](../../tensorflow/README.md#windows-cpu) | Contains TensorFlow native library for Windows |
| | [tensorflow-native-cu110(linux-x86_84)](../../tensorflow/README.md#linux-gpu) | Contains TensorFlow native library for Linux with CUDA 11.0 |
| | tensorflow-native-cu101(linux-x86_84) (deprecated) | Contains TensorFlow native library <= 2.3.0 for Linux with CUDA 10.1 |
| | | |
| [ai.djl.paddlepaddle](https://search.maven.org/search?q=g:ai.djl.paddlepaddle) | [paddlepaddle-engine](../../paddlepaddle/paddlepaddle-engine/README.md#installation) | PaddlePaddle engine adapter |
| | [paddlepaddle-model-zoo](../../paddlepaddle/paddlepaddle-model-zoo/README.md#installation) | Contains state of the art PaddlePaddle models |
| | [paddlepaddle-native-auto](../../paddlepaddle/paddlepaddle-engine/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [paddlepaddle-native-cpu(osx-x86_64)](../../paddlepaddle/paddlepaddle-engine/README.md#macos) | Contains PaddlePaddle native library for macOS |
| | [paddlepaddle-native-cpu(linux-x86_64)](../../paddlepaddle/paddlepaddle-engine/README.md#linux) | Contains PaddlePaddle native library for Linux |
| | [paddlepaddle-native-cpu(win-x86_64)](../../paddlepaddle/paddlepaddle-engine/README.md#windows) | Contains PaddlePaddle native library for Windows |
| | [paddlepaddle-native-cu102(linux-x86_64)](../../paddlepaddle/paddlepaddle-engine/README.md#Linux-GPU) | Contains PaddlePaddle native library for Linux with CUDA 10.2 |
| | [paddlepaddle-native-cu101(linux-x86_64)](../../paddlepaddle/paddlepaddle-engine/README.md#Linux-GPU) | Contains PaddlePaddle native library for Linux with CUDA 10.1 |
| | | |
| [ai.djl.onnxruntime](https://search.maven.org/search?q=g:ai.djl.onnxruntime) | [onnxruntime-engine](../../onnxruntime/onnxruntime-engine/README.md#installation) | ONNX Runtime engine adapter |
| | onnxruntime-api(deprecated) | Please use com.microsoft.onnxruntime:onnxruntime instead |
| | onnxruntime-native-auto(deprecated) | Please use com.microsoft.onnxruntime:onnxruntime instead |
| | onnxruntime-native-cpu(deprecated) | Please use com.microsoft.onnxruntime:onnxruntime instead |
| | | |
| [ai.djl.dlr](https://search.maven.org/search?q=g:ai.djl.dlr) | [dlr-engine](../../dlr/dlr-engine/README.md#installation) | Neo DLR engine adapter |
| | [dlr-native-auto](../../dlr/dlr-engine/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [dlr-native-cpu(osx-x86_64)](../../dlr/dlr-engine/README.md#macos) | Contains Neo LR native library for macOS |
| | [dlr-native-cpu(linux-x86_64)](../../dlr/dlr-engine/README.md#linux) | Contains Neo DLR native library for Linux |
| | | |
| [ai.djl.tflite](https://search.maven.org/search?q=g:ai.djl.tflite) | [tflite-engine](../../tflite/tflite-engine/README.md#installation) | TensorFlow Lite engine adapter |
| | [tflite-native-auto](../../tflite/tflite-engine/README.md#automatic-recommended) | A placeholder package to automatically download native libraries for your platform |
| | [tflite-native-cpu(osx-x86_64)](../../tflite/tflite-engine/README.md#macos) | Contains TensorFlow Lite native library for macOS |
| | [tflite-native-cpu(linux-x86_64)](../../tflite/tflite-engine/README.md#linux) | Contains TensorFlow Lite native library for Linux |
| | | |
| [ai.djl.ml.xgboost](https://search.maven.org/search?q=g:ai.djl.ml.xgboost) | [xgboost](../../ml/xgboost/README.md#installation) | XGBoost engine adapter |
| | | |
| [ai.djl.android](https://search.maven.org/search?q=g:ai.djl.android) | [core](../../android/README.md#installation) | Contains Android specific utilities (e.g. ImageFactory) for DJL |
| | [pytorch-native](../../android/README.md#installation) | Contains DJL PyTorch Android native package |
| | | |
| [ai.djl.aws](https://search.maven.org/search?q=g:ai.djl.aws) | [aws-ai](../../extensions/aws-ai/README.md#installation) | Contains classes that make it easy for DJL to access AWS services |
| | | |
| [ai.djl.fasttext](https://search.maven.org/search?q=g:ai.djl.fasttext) | [fasttext](../../extensions/fasttext/README.md#installation) | Contains NLP utility classes NLP for fastText |
| | | |
| [ai.djl.hadoop](https://search.maven.org/search?q=g:ai.djl.hadoop) | [hadoop](../../extensions/hadoop/README.md#installation) | Contains classes that make it easy for DJL access HDFS |
| | | |
| [ai.djl.sentencepiece](https://search.maven.org/search?q=g:ai.djl.sentencepiece) | [sentencepiece](../../extensions/sentencepiece/README.md#installation) | Contains NLP utility classes NLP for SentencePiece |
