# Train CAPTCHA model

In this example, you learn how to train the dataset with multiple inputs and labels.

The source code for this example can be found at [TrainCaptcha.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainCaptcha.java).

## Setup guide

To configure your development environment, follow [setup](../../docs/development/setup.md).

## Run CAPTCHA training example

### Build the project and run

The following command trains the model for two epochs. The trained model is saved in the `build/model` folder.

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainCaptcha
```

Your output should look like the following:

```text
[INFO ] - Running TrainCaptcha on: cpu(0).
[INFO ] - Load MXNet Engine Version 1.6.0 in 0.133 ms.
Training:    100% |████████████████████████████████████████| acc_digit_0: 0.56, acc_digit_1: 0.41, ..., speed: 17.10 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 0 finished.
[INFO ] - Train: acc_digit_0: 0.57, acc_digit_1: 0.46, ...
[INFO ] - Validate: acc_digit_0: 0.71, acc_digit_1: 0.67, ...
Training:    100% |████████████████████████████████████████| acc_digit_0: 1.00, acc_digit_1: 0.91, ..., speed: 18.68 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 1 finished.
[INFO ] - Train: acc_digit_0: 0.96, acc_digit_1: 0.93, ...
[INFO ] - Validate: acc_digit_0: 0.88, acc_digit_1: 0.75, ...
[INFO ] - Training: 440 batches
[INFO ] - Validation: 7 batches
[INFO ] - train P50: 1844.790 ms, P90: 1903.755 ms
[INFO ] - forward P50: 21.221 ms, P90: 21.790 ms
[INFO ] - training-metrics P50: 0.031 ms, P90: 0.039 ms
[INFO ] - backward P50: 16.837 ms, P90: 17.538 ms
[INFO ] - step P50: 19.222 ms, P90: 20.189 ms
[INFO ] - epoch P50: 843.087 s, P90: 843.087 s
```

The results show that you reached 88 percent validation accuracy at the end of the second epoch.

You can also run the example with your own arguments. For example, you can train for five epochs using batch size 64 and save the model to a specified folder `mlp_model` using the following command:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainCaptcha --args="-e 5 -b 64 -o mlp_model"
```

The following table shows the list of available arguments:


 | Argument   | Comments                                 |
 | ---------- | ---------------------------------------- |
 | `-e`       | Number of epochs to train. |
 | `-b`       | Batch size to use for training. |
 | `-g`       | Maximum number of GPUs to use. Default uses all detected GPUs. |
 | `-o`       | Directory to save the trained model. |
