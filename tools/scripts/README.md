# Start your benchmark with DJL

This is the documentation to setup benchmark for Deep Java Library (DJL) on Deep Learning Benchmark.

## `create_with_nvidia_docker.sh`
run the `create_with_nvidia_docker.sh` script in an machine to create your base environment.
Currently, we use a clean Ubuntu 16.04 build with this script.

## `build.gradle`
The `build.gradle` file is used to run inside the docker/ami. It can be used to run all examples available in DJL.

## `benchmark.sh`
This file can be used to run the benchmark on all inference examples.

## `train_benchmark.sh`
This file will execute all training tests

## Debug with MXNet
If you need to get the profiling result on the time for DJL when you use MXNet as your engine,
you can set the following environment variable to obtain the profiling report from Engine.

Basic profiling:
```
export MXNET_PROFILER_AUTOSTART=1
```

To get the break down on all operators:
```
export MXNET_PROFILER_AUTOSTART=1
export MXNET_PROFILER_MODE=15
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=0
```