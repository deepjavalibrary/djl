Joule examples
==============

This module contains example project to demonstrate how developer can use Joule API.

There are three examples:

1. Image classification example
2. Single-shot Object detection example
3. Bert question ans answer example

## Building From Source

Once you check out the code, you can build it using gradle:

```sh
cd examples
./gradlew build
```

If you want to skip unit test:
```sh
./gradlew build -x test
```

By default, Joule examples will use `mxnet-mkl` as a backend.

## Running example code locally

### Download model files
Example models can be downloaded from MXNet model zoo: <https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md>
You can download and unzip the .mar file, for example:

```
cd build
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
unzip squeezenet_v1.1.mar
```

In this example, .mar file will be download automatically if you specify the model name 
from the MXNet model zoo, e.g.: squeezenet_v1.1 or resnet50_ssd_model


### Prepare test image files:

Two test images can be found in project test resource folder: src/test/resources.
You can also down cats image from internet.


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

### Run default example - Image Classification

There is a gradle task that can run examples, by default it will run Image Classification example:

```sh
cd example
./gradlew run --args="-n squeezenet_v1.1 -i src/test/resources/kitten.jpg"
```

### Run a different example - Single-shot Object Detection

You can specify different example class with System property: "main"

```sh
cd example
./gradlew -Dmain=software.amazon.ai.example.SsdExample run --args="-n resnet50_ssd_model -l build/logs -i src/test/resources/3dogs.jpg"
```

With above command, an output image with bounding box will be save at: build/logs/ssd.jpg.
