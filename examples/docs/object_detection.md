# Object detection using model zoo model

[Object detection](https://en.wikipedia.org/wiki/Object_detection) is a computer vision technique
for locating instances of objects in images or videos.

In this example we will show you how to implement inference code with [ModelZoo model](../../docs/model-zoo.md) to detect dogs in an image.

The following is the example source code: [ObjectDetection.java](https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/ObjectDetection.java).

You can also find jupyter notebook tutorial [here](../../jupyter/README.md#run-object-detection-with-model-zoo).
The jupyter notebook will explain the key concept in detail.

## Setup Guide

Please follow [setup](../../docs/development/setup.md) to configure your development environment.

## Run object detection example

### Input image file
You can find the image used in this example in project test resource folder: `src/test/resources/3dogs.jpg`

![dogs](../src/test/resources/3dogs.jpg)

### Build the project and run

```
cd examples
./gradlew run -Dmain=ai.djl.examples.inference.ObjectDetection
```

```text
[INFO ] - Detected objects image has been saved in: build/output/ssd.jpg
[INFO ] - [
        class: "dog", probability: 0.99839, bounds: [x=0.615, y=0.312, width=0.281, height=0.381]
        class: "dog", probability: 0.98797, bounds: [x=0.455, y=0.468, width=0.183, height=0.205]
        class: "dog", probability: 0.66150, bounds: [x=0.310, y=0.207, width=0.193, height=0.448]
]
```

With the previous command, an output image with bounding box will be saved at: build/output/ssd.jpg:

![detected-dogs](img/detected-dogs.jpg)
