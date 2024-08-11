# Train CIFAR-10 Dataset using ResNet50


In this example, you learn how to train the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset with Deep Java Library (DJL) using [Transfer Learning](https://en.wikipedia.org/wiki/Transfer_learning).

You can find the example source code in: [TrainResnetWithCifar10.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/transferlearning/TrainResnetWithCifar10.java).

You can also find the Jupyter notebook tutorial [here](https://docs.djl.ai/master/docs/demos/jupyter/transfer_learning_on_cifar10.html).
The Jupyter notebook explains the key concepts in detail.

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

## Run CIFAR-10 training using ResNet50

The models you use are available in the [DJL Model Zoo](../../model-zoo/README.md) and [MXNet Model Zoo](../../engines/mxnet/mxnet-model-zoo/README.md). 
We can simply load and use them as follows:

### Using a DJL model from Model Zoo

A DJL model is natively implemented using our Java API. It's defined using the Block API.
Import the `ai.djl.basicmodelzoo.cv.classification.ResNetV1` class and use its builder to specify various configurations such as input shape, number of layers, and number of outputs.
You can set the number of layers to create variants of [ResNet](https://en.wikipedia.org/wiki/Residual_neural_network) such as ResNet18, ResNet50, and ResNet152.

For example, you can create ResNet50 using the following code:

```java
 Block resNet50 = new ResNetV1.Builder()
                        .setImageShape(new Shape(3, 32, 32))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();
```

To run the example, use the following command: 

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-e 10 -b 32 -g 1"
```

You can use the option `-p` to specify pre-trained parameters. 

### Using a MXNet model from the MXNet Model Zoo

A MXNet model is pre-trained using the [Apache MXNet(incubating)](https://mxnet.incubator.apache.org/) deep learning library and [Gluon CV](https://gluon-cv.mxnet.io/) computer vision toolkit.
Models are trained in Python and exported to `.symbol`(model architecture) and `.params`(trained parameter values) files. These models are also known as symbolic models.

To run the example using MXNet model, use the option `-s` as shown in the following command: 

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-e 10 -b 32 -g 1 -s -p"
```

You can also remove the option `-p` to train from scratch.
It will still use the exported MXNet model architecture, but will re-initialize parameters with random values to train from scratch.


## Learning Rate Schedule
Learning rate is one of the most important [hyperparameters](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) in deep learning.
It's part of the optimization algorithm that controls how fast to move towards reducing your loss/objective function. 

During the training process, you should usually reduce the learning rate periodically to prevent the model from plateauing. 
You will also need different learning rate strategies based on whether you are using a pre-trained model or training from scratch.
DJL provides several built-in `Tracker`s to suit your needs. For more information, see the
[documentation](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/tracker/Tracker.html).

Here, you use a [`MultiFactorTracker`](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/training/tracker/MultiFactorTracker.html),
which allows you to reduce the learning rate after a specified number of periods.
We use a base learning rate of `0.001`, and reduce it by `sqrt(0.1)` every specified number of epochs. 
For a pre-trained model, you reduce the learning rate at the 2nd, 5th, and 8th epoch because it take less time to train and converge. 
For training from scratch, you reduce the learning rate at 20th, 60th, 90th, 120th, and 180th epoch.
 

## Train using Multiple GPUs
Using multiple GPUs can significantly increase training speed. Use the following steps to run this example using a multi-GPU machine.

### Setup a machine with multiple Nvidia GPU cards.
DJL only works with Nvidia GPUs. You need to install [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and  [cuDNN Library](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
for fast computation acceleration.
We recommend using [AWS EC2 P3](https://aws.amazon.com/ec2/instance-types/p3/) instances together with [AWS Deep Learning AMIs](https://aws.amazon.com/machine-learning/amis/) or [AWS Deep Learning Containers](https://aws.amazon.com/machine-learning/containers/).
They come with powerful Nvidia GPUs, and include pre-installed drivers and all dependent libraries.

For example, on an [p3.16xlarge](https://aws.amazon.com/ec2/instance-types/) instance with [Ubuntu Deep Learning Base AMI](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-Deep-Learning-Base-AMI-Amazon-/B077GFM7L7), 
run the following command to check the GPU status, driver information, and CUDA version.

```sh
nvidia-smi
```

You should see the following output:

```aidl
hu Nov 21 00:58:29 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.87.00    Driver Version: 418.87.00    CUDA Version: 10.1     |
|-------------------------------|----------------------|----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:17.0 Off |                    0 |
| N/A   42C    P0    45W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   1  Tesla V100-SXM2...  On   | 00000000:00:18.0 Off |                    0 |
| N/A   44C    P0    47W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   2  Tesla V100-SXM2...  On   | 00000000:00:19.0 Off |                    0 |
| N/A   45C    P0    44W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   3  Tesla V100-SXM2...  On   | 00000000:00:1A.0 Off |                    0 |
| N/A   42C    P0    41W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   4  Tesla V100-SXM2...  On   | 00000000:00:1B.0 Off |                    0 |
| N/A   42C    P0    43W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   5  Tesla V100-SXM2...  On   | 00000000:00:1C.0 Off |                    0 |
| N/A   43C    P0    42W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   6  Tesla V100-SXM2...  On   | 00000000:00:1D.0 Off |                    0 |
| N/A   42C    P0    43W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+
|   7  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   44C    P0    43W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------|----------------------|----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```


### Run this example with multi-GPU
Use the option `-g` to specify how many GPUs to use, and use `-b` to specify the batch size. 
Usually, you use `32*number_of_gpus`, so each GPU will get a data batch size of 32. For 4 GPUs, the total batch size is 128.

Run the following command to train using 4 GPUs:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-e 10 -b 128 -g 4 -p"
```

You should see the following output:

```bash
> Task :examples:run
[INFO ] - Running TrainResnetWithCifar10 on: 4 GPUs, epoch: 10.
[INFO ] - Load library 1.5.0 in 0.225 ms.
Loading:     100% |████████████████████████████████████████|
[00:06:57] src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:97: Running performance tests to find the best convolution algorithm, this can take a while... (set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
Training:    100% |████████████████████████████████████████| accuracy: 0.51 loss: 1.39 speed: 527.67 images/sec
Validating:  100% |████████████████████████████████████████|
[00:10:01] src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:97: Running performance tests to find the best convolution algorithm, this can take a while... (set the environment variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)
[INFO ] - Epoch 0 finished.
[INFO ] - train accuracy: 0.50522, train loss: 1.392299
[INFO ] - validate accuracy: 0.5627, validate loss: 1.226838
```


The following is the list of available arguments for this example:

 | Argument   | Comments                                 |
 | ---------- | ---------------------------------------- |
 | `-e`       | Number of epochs to train. |
 | `-b`       | Batch size to use for training. |
 | `-g`       | Maximum number of GPUs to use. Default will use all detected GPUs. |
 | `-o`       | Directory to save the trained model. |
 | `-s`       | Use symbolic ResNet50V1 from MXNet model zoo |
 | `-p`       | Use model with pre-trained parameter weights |
 | `-m`       | Only train a fixed number of batches each epoch(for debug and test) |
 | `-d`       | Model directory to load the model checkpoint and continue training |
 | `-r`       | Criteria to use for selecting model from model zoo |
