Joule examples
==============

This module contains example project to demonstrate how developer can use Joule API.

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

## Running example code locally

###
Install MXNet 1.5, current Joule only works with MXNet 1.5 release.


```
sudo pip install mxnet-mkl --pre
```

### Download model files
Example models can be downloaded from MXNet model zoo: <https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md>
You have to unzip the .mar file, for example:

```
cd build
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
unzip squeezenet_v1.1.mar
```

### Download image file:
A kitten image can be found: <https://s3.amazonaws.com/model-server/inputs/kitten.jpg>

```
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```

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

### Run default example
```sh
cd example
./gradlew run --args="-p build/ -n squeezenet_v1.1 -c 1 -l build/logs -i build/kitten.jpg"
```

### Run a different example

You can specify different example class with System property: "main"

```sh
cd example
./gradlew -Dmain=com.amazon.ai.example.SsdExample run --args="-p build/ -n squeezenet_v1.1 -c 1 -l build/logs -i build/kitten.jpg"
```


