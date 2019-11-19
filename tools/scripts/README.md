# Start your benchmark with DJL

This topic explains how to set up a benchmark for Deep Java Library (DJL) by using Deep Learning Benchmark.
## `create_with_nvidia_docker.sh`
Run the `create_with_nvidia_docker.sh` script on your machine to create your base environment.

## `build.gradle`
The `build.gradle` file should be run inside the docker/ami folder. It can be used to run all examples available in DJL.

## `benchmark.sh`
The `benchmark.sh` file can be used to run the benchmark on all inference examples.

## `train_benchmark.sh`
The `train_benchmark.sh` file can be used to execute all training tests.

## Debug with MXNet
To get the profiling result when you use MXNet as your engine, set the following environment variables to obtain the profiling report.

For basic profiling, set the following environment variable:
```
export MXNET_PROFILER_AUTOSTART=1
```

To get the break down on all operators, set the following environment variables:
```
export MXNET_PROFILER_AUTOSTART=1
export MXNET_PROFILER_MODE=15
export MXNET_EXEC_BULK_EXEC_INFERENCE=0
export MXNET_EXEC_BULK_EXEC_TRAIN=0
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN=0
```