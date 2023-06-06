# Imperative Object Detection example - Pikachu Dataset

[Object detection](https://en.wikipedia.org/wiki/Object_detection) is a computer vision technique
for locating instances of objects in images or videos. In this example, you can find an imperative implemention of an 
SSD model, and the way to train it using the Pikachu Dataset. The code for the example can be found in 
[TrainPikachu.java](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/training/TrainPikachu.java). 
The code for the implementation of SSD can be found in [SingleShotDetection.java](https://github.com/deepjavalibrary/djl/blob/master/model-zoo/src/main/java/ai/djl/basicmodelzoo/cv/object_detection/ssd/SingleShotDetection.java).

There are no small datasets, like MNIST or Fashion-MNIST, in the object detection field. In order to quickly test models,
you are using a small dataset of Pikachu images. It contains a series of background images on which a Pikachu image
is placed at a random position. The Pikachu images are also generated in different angles and sizes.  

## Setup guide

Follow [setup](../../docs/development/setup.md) to configure your development environment.

## Run SSD training example

### Build the project and run it
The following command trains the model for 2 epochs. The trained model is saved in the following folder: `build/model`.

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainPikachu
```

Your output should look like the following:

```text
[INFO ] - Running TrainPikachu on: cpu(0), epoch: 2.
[INFO ] - Load library 1.5.0 in 0.183 ms.
Training:    100% |████████████████████████████████████████| loss: 4.991e-02f, classAccuracy: 0.9792, bboxError: 6.094e-04, speed: 7.37 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 0 finished.
[INFO ] - train loss: 0.04971575, train class accuracy: 0.9792732, train bounding box error: 6.076591E-4
[INFO ] - validate loss: 0.021246385, validate class accuracy: 0.9993332, validate bounding box error: 5.937452E-4
Training:    100% |████████████████████████████████████████| loss: 6.521e-03f, classAccuracy: 0.9996, bboxError: 4.977e-04, speed: 7.38 images/sec
Validating:  100% |████████████████████████████████████████|
[INFO ] - Epoch 1 finished.
[INFO ] - train loss: 0.006508477, train class accuracy: 0.9995894, train bounding box error: 4.9679517E-4
[INFO ] - validate loss: 0.005975074, validate class accuracy: 0.9995536, validate bounding box error: 5.251629E-4
[INFO ] - Training: 28 batches
[INFO ] - Validation: 3 batches
[INFO ] - train P50: 4349.508 ms, P90: 4909.701 ms
[INFO ] - forward P50: 6.207 ms, P90: 10.621 ms
[INFO ] - training-metrics P50: 3972.653 ms, P90: 4364.357 ms
[INFO ] - backward P50: 5.000 ms, P90: 9.078 ms
[INFO ] - step P50: 4.436 ms, P90: 6.829 ms
[INFO ] - epoch P50: 138.332 s, P90: 138.332 s
```

You can also run the example with your own arguments, for example, to train 5 epochs using batch size 64, and save it to a specified folder `ssd_model`:

```sh
cd examples
./gradlew run -Dmain=ai.djl.examples.training.TrainPikachu --args="-e 5 -b 64 -o ssd_model"
```

The following table shows the list of available arguments:

 | Argument   | Comments                                 |
 | ---------- | ---------------------------------------- |
 | `-e`       | Number of epochs to train. |
 | `-b`       | Batch size to use for training. |
 | `-g`       | Maximum number of GPUs to use. Default uses all detected GPUs. |
 | `-o`       | Directory to save the trained model. |
 
 
### Run prediction

There is a `predict` method available in the `TrainPikachu` class.
Just pass the directory of saved models and the path to the image for prediction.
For example:

```java
TrainPikachu trainPikachu = new TrainPikachu();
trainPikachu.predict("build/model", "src/test/resources/pikachu.jpg");
```
