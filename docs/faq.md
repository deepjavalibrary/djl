# FAQ

### Why Deep Java Library (DJL)?

- Prioritizes the Java developer's experience
- Makes it easy for new machine learning developers to get started
- Allows developers to write modular, reusable code
- Reduces friction for deploying to a production environment
- Connects model developers with their consumers using the model zoo
- Allows developers to write code once and run it on any deep learning engine
- Allows developers to use engine specific features

### Which models can I run with DJL?
While DJL is designed to be engine-agnostic and can easily integrated with any engines, we currently
support the following models:

- PyTorch TorchScript model
- TensorFlow SavedModel bundle
- Apache MXNet model
- ONNX model
- TensorRT model
- Python script model
- TFLite model
- XGBoost model
- LightGBM model
- Sentencepiece model
- fastText/BlazingText model

### Does DJL support inference on GPU?
Yes. DJL does support inference on GPU. If GPUs are available, DJL automatically detects the GPU,
and runs inference on a single GPU by default.

#### How to run GPU inference for OnnxRuntime
OnnxRuntime engine by default depends on `com.microsoft.onnxruntime:onnxruntime` CPU package.
You need install `com.microsoft.onnxruntime:onnxruntime_gpu` to enable GPU for OnnxRuntime. 
See: [Install Onnxruntime GPU package](../engines/onnxruntime/onnxruntime-engine/README.md#install-gpu-package)

### Does DJL support inference on multiple threads?
Yes. DJL offers high performance multi-threaded inference. For more information, see the
[inference_performance_optimization](development/inference_performance_optimization.md).

### Does DJL support training on GPU?
Yes. DJL offers multi-GPU support. DJL can automatically detect if GPUs are available. If GPUs are available, it will
run on a single GPU by default, unless the user specifies otherwise.

During training, if you wish to train on multiple GPUs or if you wish to limit the number of GPUs to be used (you may want to limit the number of GPU for smaller datasets), you have to configure the `TrainingConfig` to do so by
setting the devices. For example, if you have 7 GPUs available, and you want the `Trainer` to train on 5 GPUs, you can configure it as follows. 

```java
    int maxNumberOfGpus = 5;
    TrainingConfig config = new DefaultTrainingConfig(initializer, loss)
            .setOptimizer(optimizer)
            .addEvaluator(accuracy)
            .setBatchSize(batchSize)
            // Set the devices to run on multi-GPU
            .setDevices(Engine.getInstance().getDevices(numberOfGpus));
```

All of the examples in the example folder can be run on 
multiple GPUs with the appropriate arguments. Follow the steps in the example to
[train a ResNet50 model on CIFAR-10 dataset](https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_cifar10_resnet.md#train-using-multiple-gpus) on a GPU.

#### Does DJL support distributed training?
DJL does not currently support distributed training.

### Can I run DJL on other versions of PyTorch?
Yes. [Each DJL release has a range of PyTorch version](../engines/pytorch/pytorch-engine/README.md#supported-pytorch-versions).
You can set `PYTORCH_VERSION` environment variable or Java System properties to choose
a different version of PyTorch.

If you want to load your custom build PyTorch you can follow [this instruction](../engines/pytorch/pytorch-engine/README.md#load-your-own-pytorch-native-library).

### How can I pass arbitrary input data type to a PyTorch model? 
DJL uses `NDList` as a standard data type to pass to the model. `NDList` is a flat list of tensor.
A typical PyTorch model can accept a Map, List or Tuple of tensor. DJL provides the following way
to automatically map `NDList` to PyTorch's `IValue`:

1. set each `NDArray` name with suffix `[]` to group them into a `list[Tensor]`, see: [example](https://github.com/deepjavalibrary/djl/blob/master/engines/pytorch/pytorch-engine/src/test/java/ai/djl/pytorch/jni/IValueUtilsTest.java#L79)
2. set each `NDArray` name with suffix `()` to group them into a `tuple[Tensor]`, see: [example](https://github.com/deepjavalibrary/djl/blob/master/engines/pytorch/pytorch-engine/src/test/java/ai/djl/pytorch/jni/IValueUtilsTest.java#L29)
3. set each `NDArray` name with suffix `group.key` to group them into a `dict(str, Tensor)`, see: [example](https://github.com/deepjavalibrary/djl/blob/master/engines/pytorch/pytorch-engine/src/test/java/ai/djl/pytorch/jni/IValueUtilsTest.java#L51)

If your model requires non-tensor input or complex `IValue`, you have to use `IValue` class directly
(This makes your code bound to PyTorch engine). See [this example](https://github.com/deepjavalibrary/djl/blob/master/engines/pytorch/pytorch-engine/src/test/java/ai/djl/pytorch/integration/IValueTest.java).

### How can I do pre/post processing with OnnxRuntime?
DJL provides a [hybrid engine](hybrid_engine.md) design that allows you to leverage PyTorch/MXNet/TensorFlow
to performance `NDArray` operations for OnnxRuntime.

### How can I run Python model in DJL?
Yes, DJL has [Python engine](https://github.com/deepjavalibrary/djl-serving/tree/master/engines/python)
that allows you run inference with Python code. The Python engine provides the same code experience
as other engines, and makes it easy for you to migrate to native Java model easier in future.

#### How can I run Python based pre/post processing?
Sometime, it's hard to port python data processing code into java due to lack of equivalent java
library or implementing it in Java is time consuming. In this case, you can package your Python
code as a processing model. See [How to run python pre/post processing](https://github.com/deepjavalibrary/djl-demo/tree/master/development/python)

### How can I get help if I run into problems?
You can check out our [troubleshooting document](development/troubleshooting.md),
[discussions](https://github.com/deepjavalibrary/djl/discussions),
[issues](https://github.com/deepjavalibrary/djl/issues).

You can also join our [<img src='https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_slack-512.png' width='20px' /> slack channel](http://tiny.cc/djl_slack)
to get in touch with the development team, for questions and discussions.
