# FAQ

### 1.  Why Deep Java Library (DJL)?

- Prioritizes the Java developer's experience
- Makes it easy for new machine learning developers to get started
- Allows developers to write modular, reusable code
- Reduces friction for deploying to a production environment
- Connects model developers with their consumers using the model zoo
- Allows developers to write code once and run it on any deep learning engine
- Allows developers to use engine specific features

### 2. Which models can I run with DJL?
While DJL is designed to be engine-agnostic and can easily integrated with any engines, we currently
support the following models:

- PyTorch TorchScript model
- TensorFlow SavedModel bundle
- Apache MXNet model
- ONNX model
- TensorRT model
- Python script model
- PaddlePaddle model
- TFLite model
- Neo DLR (TVM) model
- XGBoost model
- LightGBM model
- Sentencepiece model
- fastText/BlazingText model

### 3. Does DJL support inference on GPU?
Yes. DJL does support inference on GPU. If GPUs are available, DJL automatically detects the GPU, and runs inference on a single GPU by default.

### 4. Does DJL support training on GPU?
Yes. DJL offers multi-GPU support. DJL can automatically detect if GPUs are available. If GPUs are available, it will
run on a single GPU by default, unless the user specifies otherwise.

During training, if you wish to train on multiple GPUs or if you wish to limit the number of GPUs to be used (you may want to limit the number of GPU for smaller datasets), you have to configure the `TrainingConfig` to do so by
setting the devices. For example, if you have 7 GPUs available, and you want the `Trainer` to train on 5 GPUs, you can configure it as follows. 

    int maxNumberOfGpus = 5;
    TrainingConfig config = new DefaultTrainingConfig(initializer, loss)
            .setOptimizer(optimizer)
            .addEvaluator(accuracy)
            .setBatchSize(batchSize)
            // Set the devices to run on multi-GPU
            .setDevices(Engine.getInstance().getDevices(numberOfGpus));
All of the examples in the example folder can be run on 
multiple GPUs with the appropriate arguments. Follow the steps in the example to [train a ResNet50 model on CIFAR-10 dataset](https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_cifar10_resnet.md#train-using-multiple-gpus) on a GPU.

### 5. Does DJL support inference on multiple threads?
Yes. DJL offers multi-threaded inference. If using the MXNet engine for a multi-threaded inference case, you need to 
specify the 'MXNET_ENGINE_TYPE' environment variable to 'NaiveEngine'. For more information, see the
[inference_performance_optimization](development/inference_performance_optimization.md).

### 6. Does DJL support distributed training?
DJL does not currently support distributed training.

### 7. Can I run DJL on other versions of PyTorch?
Yes. [Each DJL release has a range of PyTorch version](../engines/pytorch/pytorch-engine/README.md#installation).
You can set set `PYTORCH_VERSION` environment vairable or Java System properties to choose
a different version of PyTorch.

If you want to load your custom build PyTorch you can follow [this instruction](../engines/pytorch/pytorch-engine/README.md#load-your-own-pytorch-native-library).
