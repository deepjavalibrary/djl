Single-shot Object detection
============================

In this tutorial, you walk through the Single-shot Object detection model trained by MXNet.

## Setup Guide

#### Prepare test image files

You can find the test image in the project test resource folder: src/test/resources/3dogs.jpg.
You can also download other images from the internet.


## Build example project

### gradle build

```sh
cd examples
./gradlew build
```

### maven build

```sh
cd examples
./mvnw package
```

## Run example

### Get command line parameters help
```sh
cd example
./gradlew run

>>>
>>>usage:
>>> -c,--iteration <ITERATION>     Number of iterations in each test.
>>> -d,--duration <DURATION>       Duration of the test.
>>> -i,--image <IMAGE>             Image file.
>>> -l,--log-dir <LOG-DIR>         Directory for output logs.
>>> -n,--model-name <MODEL-NAME>   Model name prefix.
>>> -p,--model-dir <MODEL-DIR>     Path to the model directory.
```

### Run example with parameters

To execute the SSD example, you need to specify a different Main class with System property: "main":

```sh
cd example
./gradlew -Dmain=software.amazon.ai.example.SsdExample run --args="-n resnet50_ssd_model -l build/logs -i src/test/resources/3dogs.jpg"
```

With the previous command, an output image with bounding box will be saved at: build/logs/ssd.jpg.
