# Benchmark your DL model

DJL offers a comprehensive script to benchmark the model on all different 
platforms for single-thread/multi-thread inference performance.This document will guide you how to run benchmark with DJL.

## Prerequisite
Please ensure Java 8+ is installed and you are using an OS that DJL supported with.

After that, you need to clone the djl project and `cd` into the folder.

DJL supported OS:

- Ubuntu 16.04 and above
- CentOS 7 (Amazon Linux 2) and above
- MacOS latest version
- Windows 10 (Windows Server 2016+)

If you are trying to use GPU, please ensure the CUDA driver is installed. You can verify that through:

```
nvcc -V
```
to checkout the version. For different Deep Learning engine you are trying to run the benchmark,
they have different CUDA version to support. Please check the individual Engine documentation to ensure your CUDA version is supported.

## Sample benchmark script

Here is a few sample benchmark script for you to refer. You can also skip this and directly follow
the 4-step instructions for your own model.

Benchmark on a Tensorflow model from http url with all-ones NDArray input for 10 times:

```
./gradlew benchmark -Dai.djl.default_engine=TensorFlow -Dai.djl.repository.zoo.location=https://storage.googleapis.com/tfhub-modules/tensorflow/resnet_50/classification/1.tar.gz?artifact_id=tf_resnet --args='-n tf_resnet -c 10 -s 1,224,224,3'
```

Similarly, this is for PyTorch

```
./gradlew benchmark -Dai.djl.default_engine=PyTorch -Dai.djl.repository.zoo.location=https://alpha-djl-demos.s3.amazonaws.com/model/djl-blockrunner/pytorch_resnet18.zip?model_name=traced_resnet18 --args='-c 10 -s 1,3,224,224'
```

### Benchmark from ModelZoo

#### MXNet

Resnet50 image classification model:

```
./gradlew benchmark --args="-c 1 -s 1,3,224,224 -n ai.djl.mxnet:resnet -r {'layers':'50','flavor':'v2','dataset':'imagenet'}"
```

#### PyTorch

SSD object detection model:

```
./gradlew benchmark -Dai.djl.default_engine=PyTorch --args="-c 1 -s 1,3,300,300 -n ai.djl.pytorch:ssd -r {'size':'300','backbone':'resnet50'}"
```


## Configuration of Benchmark script

To start your benchmarking, we need to make sure we provide the following information.

- The Deep Learning Engine
- The source of the model
- How many runs you would like to make
- Sample input for the model
- (Optional) Multi-thread benchmark

The benchmark script located [here](https://github.com/deepjavalibrary/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/benchmark/Benchmark.java).

Just do the following:

```
./gradlew benchmark --args="--help"
```

or Windows:

```
..\gradlew.bat benchmark --args="--help"
```

This will print out the possible arguments to pass in:

```
usage: ./gradlew benchmark --args='[OPTIONS]'
 -c,--iteration <ITERATION>         Number of total iterations (per thread).
 -d,--duration <DURATION>           Duration of the test in minutes.
 -h,--help                          Print this help.
 -i,--image <IMAGE>                 Image file path for benchmarking CV model.
 -l,--delay <DELAY>                 Delay of incremental threads.
 -n,--artifact-id <ARTIFACT-ID>     Model artifact id.
 -o,--output-dir <OUTPUT-DIR>       Directory for output logs.
 -r,--criteria <CRITERIA>           The criteria (json string) used for searching the model.
 -s,--input-shapes <INPUT-SHAPES>   Input data shapes for non-CV model.
 -t,--threads <NUMBER_THREADS>      Number of inference threads.
```

### Step 1: Pick your deep engine

By default, the above script will use MXNet as the default Engine, but you can always change that by adding the followings:

```
-Dai.djl.default_engine=TensorFlow # tensorflow
-Dai.djl.default_engine=PyTorch # pytorch
```
to change your default engine.

### Step 2: Identify the source of your model

DJL accept variety of models came from different places.

#### Remote location

The following is a pytorch model

```
-Dai.djl.repository.zoo.location=https://alpha-djl-demos.s3.amazonaws.com/model/djl-blockrunner/pytorch_resnet18.zip?model_name=traced_resnet18
```
We would recommend to make model files in a zip for better file tracking.

#### Local directory

Mac/Linux

```
-Dai.djl.repository.zoo.location=file:///pytorch_resnet18.zip?model_name=traced_resnet18
```

Windows

```
-Dai.djl.repository.zoo.location=file:///C:/pytorch_resnet18.zip?model_name=traced_resnet18 # windows
```

#### DJL Model zoo

You can run `listmodels` to list available models that you can use from different model zoos.

```
./gradlew listmodels # MXNet models
./gradlew listmodels -Dai.djl.default_engine=TensorFlow # TensorFlow models
./gradlew listmodels -Dai.djl.default_engine=PyTorch # PyTorch models
```

After that, just simply copy the json formatted criteria like `{"layers":"18","flavor":"v1","dataset":"imagenet"}` with the model name like `ai.djl.mxnet:resnet:0.0.1`.
Then, you can just pass These information in the `--args` (remove `0.0.1` at the end):

```
-n ai.djl.mxnet:resnet -r {"layers":"18","flavor":"v1","dataset":"imagenet"}
```

### Step 3: Define how many runs you would like to make

add `-c` inside with a number

```
-c 1000
```

This will run 1000 times inference.

### Step 4: Define your model inputs

The benchmark script support dummy NDArray inputs.
It will make fake NDArrays (like `NDArray.ones`) to feed in the model for inference.

If we would like to fake an image:

```
-s 1,3,224,224
```

This will create a NDArray of shape(1, 3, 224, 224).

If your model requires multiple inputs like three NDArrays with shape 1, 384 and 384. You can do the followings:

```
-s (1),(384),(384)
```

### Optional Step: multithreading inference 

You can also do multi-threading inference with DJL. For example, if you would like to run the inference with 10 threads:

```
-t 10
```
Best thread number for your system: The same number of cores your system have or double of the total cores.

You can also add `-l` to simulate the increment load for your inference server. It will add threads with the delay of time.

```
-t 10 -l 100
```

The above code will create 10 threads with the wait time of 100ms.

## Advanced use cases

For different purposes, we designed different mode you can play with. Such as the following arg:

```
-d 1440
```

This will ask the benchmark script repeatly running the designed task for 1440 minutes (24 hour).
If you would like to make sure DJL is stable in the long run, you can do that.

You can also keep monitoring the DJL memory usages by enable the following flag:

```
-Dcollect-memory=true
```

The memory report will be made available in `build/memory.log`.
