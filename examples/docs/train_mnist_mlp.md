# Train Handwritten Digit Recognition using Multilayer Perceptron (MLP) model

Training a model on a handwritten digit dataset, such as ([MNIST](http://yann.lecun.com/exdb/mnist/)) is like the "Hello World!" program of the deep learning world.

In this example, you learn how to train the MNIST dataset with Deep Java Library (DJL) to recognize handwritten digits in an image.

The source code for this example can be found at [TrainMnist.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainMnist.java).

You can also use the [Jupyter notebook tutorial](https://docs.djl.ai/master/docs/demos/jupyter/tutorial/02_train_your_first_model.html).
The Jupyter notebook explains the key concepts in detail.

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run handwritten digit recognition example

### Build the project and run

The following command trains the model for two epochs. The trained model is saved in the `build/model` folder.

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainMnist
```

Your output should look like the following:

```text
[INFO ] - Running TrainMnist on: cpu(0), epoch: 2.
[INFO ] - Load library 1.5.0 in 0.121 ms.
Training:    100% |████████████████████████████████████████| accuracy: 0.89 loss: 0.38 speed: 0.00 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 0 finished.
[INFO ] - train accuracy: 0.8871, train loss: 0.38165984
[INFO ] - validate accuracy: 0.9245, validate loss: 0.25397184
Training:    100% |████████████████████████████████████████| accuracy: 0.96 loss: 0.12 speed: 0.00 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 1 finished.
[INFO ] - train accuracy: 0.96363336, train loss: 0.12292298
[INFO ] - validate accuracy: 0.9693, validate loss: 0.099014595
[INFO ] - Training: 1875 batches
[INFO ] - Validation: 312 batches
[INFO ] - train P50: 10.546 ms, P90: 14.872 ms
[INFO ] - forward P50: 0.370 ms, P90: 0.495 ms
[INFO ] - training-metrics P50: 0.969 ms, P90: 2.148 ms
[INFO ] - backward P50: 0.702 ms, P90: 1.018 ms
[INFO ] - step P50: 0.394 ms, P90: 0.585 ms
[INFO ] - epoch P50: 29.520 s, P90: 29.520 s
```

The results show that you reached 96.93 percent validation accuracy at the end of the second epoch.


You can also run the example with your own arguments. For example, you can train for five epochs using batch size 64 and save the model to a specified folder `mlp_model` using the following command:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainMnist --args="-e 5 -b 64 -o mlp_model"
```

The following table shows the list of available arguments:


 | Argument   | Comments                                 |
 | ---------- | ---------------------------------------- |
 | `-e`       | Number of epochs to train. |
 | `-b`       | Batch size to use for training. |
 | `-g`       | Maximum number of GPUs to use. Default uses all detected GPUs. |
 | `-o`       | Directory to save the trained model. |
