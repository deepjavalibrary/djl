# Multi-threaded inference with DJL

Multi-threaded inference is supported by Deep Java Library (DJL).
It can help to increase the throughput of your inference on multi-core CPUs and GPUs.

## Overview

DJL predictor is not thread-safe.
You must create a new [Predictor](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/Predictor.html) for each thread.

For a reference implementation, see [Multi-threaded Benchmark](https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/benchmark/MultithreadedBenchmark.java).

you need to set corresponding configuration based on the engine you want to use.

## Apache MXNet

## Engine configuration
To use Apache MXNet Engine to run multi-threading, complete the following steps.

## Enable NaiveEngine with Apache MXNet
If using the MXNet engine for a multi-threaded inference case, you need to specify the 'MXNET_ENGINE_TYPE' environment variable using the following command:

```
export MXNET_ENGINE_TYPE=NaiveEngine
```

To get the best throughput, you may also need to set 'OMP_NUM_THREADS' environment variable:

```
export OMP_NUM_THREADS=1
```

## PyTorch

Currently, PyTorch engine doesn't support multithreading inference. You may see random crash during the inference. 
We expect to fix the issue in the future release.

You might also want to check out [how_to_optimize_inference_performance](../../docs/pytorch/how_to_optimize_inference_performance.md)
to optimize the inference performance.
