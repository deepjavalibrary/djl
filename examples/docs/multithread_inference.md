# Multi-threaded inference with DJL

Multi-threaded inference is supported by Deep Java Library (DJL).
It can help to increase the throughput of your inference on multi-core CPUs and GPUs.

## Overview

DJL predictor is not thread-safe.
You must create a new [Predictor](https://javadoc.djl.ai/api/0.2.0/index.html?ai/djl/inference/Predictor.html) for each thread.

For a reference implementation, see [Multi-threaded Benchmark](../src/main/java/ai/djl/examples/inference/MultithreadedBenchmark.java).

## MXNet Engine configuration
To use MXNet Engine to run multi-threading, complete the following steps.

### Enable NaiveEngine with MXNet
If using the MXNet engine for a multi-threaded inference case, you need to specify the 'MXNET_ENGINE_TYPE' environment variable using the following command:```
```
export MXNET_ENGINE_TYPE=NaiveEngine
```

### Save your inference memory with thread-safe mode (Experimental)

This is an experimental feature used in MXNet to share the parameters' memory across all predictors.
Memory consumption will not change if you change the number of threads.
By default, the parameters for the model will be copied for each thread.

Please add the following parameter to your Java application:
```
-DMXNET_THREAD_SAFE_INFERENCE=true
```
