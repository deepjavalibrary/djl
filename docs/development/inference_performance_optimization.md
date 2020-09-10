# Inference Performance Optimization with DJL

The document covers several tricks of how you can tune your inference performance based on the engine you use 
including multithreading support, engine threads configuration and how to enable DNNL(MKLDNN).
 
## Multithreading Support

Multi-threaded inference is supported by Deep Java Library (DJL).
It can help to increase the throughput of your inference on multi-core CPUs and GPUs.

DJL `Predictor` is not thread-safe, so we recommend creating a new [Predictor](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/Predictor.html) for each thread.

For a reference implementation, see [Multi-threaded Benchmark](https://github.com/awslabs/djl/blob/master/examples/src/main/java/ai/djl/examples/inference/benchmark/MultithreadedBenchmark.java).

you need to set corresponding configuration based on the engine you want to use.

### Apache MXNet

#### Engine configuration
To use Apache MXNet Engine to run multi-threading, complete the following steps.

#### Enable NaiveEngine with Apache MXNet
If using the MXNet engine for a multi-threaded inference case, you need to specify the 'MXNET_ENGINE_TYPE' environment variable using the following command:

```
export MXNET_ENGINE_TYPE=NaiveEngine
```

To get the best throughput, you may also need to set 'OMP_NUM_THREADS' environment variable:

```
export OMP_NUM_THREADS=1
```

DJL MXNet by default would copy parameters of the model for each Predictor you created.
To save the memory, you might want to use experimental thread-safe predictor by adding VM option shown below.

```
-Dai.djl.mxnet.use_thread_safe_predictor=true
```

There are several limitations on this experimental feature.
1. It doesn't support the sparse format.
2. The underlying CachedOp don't support backward(), which means you can't do transfer learning.
3. Only symbolic model is supported.

You can find more information on [thread-safe CachedOp limitations](https://github.com/apache/incubator-mxnet/blob/master/docs/static_site/src/pages/api/cpp/docs/tutorials/multi_threaded_inference.md#current-limitations)

### PyTorch

### Multithreading Inference
To use multithreading inference feature, we have to disable GC to close the NDArray by

```
# If you are using DJL 0.5.0
-Dai.djl.pytorch.disable_close_resource_on_finalize=true
# If you are using DJL 0.6.0
-Dai.djl.disable_close_resource_on_finalize=true
```

Please make sure all the NDArrays are attached to the NDManager.
It is expected to be fixed in the future.

### oneDNN(MKLDNN) acceleration
Unlike TensorFlow and Apache MXNet, PyTorch by default doesn't enable MKLDNN which is treated as a device type like CPU and GPU.
You can enable it by

```
-Dai.djl.pytorch.use_mkldnn=true
```

You might see the exception if certain data type or operator is not supported with the oneDNN device.

### Thread configuration
There are two configurations you can set to optimize the inference performance.

```
-Dai.djl.pytorch.num_interop_threads=[num of the interop threads]
```

It configures the number of the operations JIT interpreter fork to execute in parallel.

```
-Dai.djl.pytorch.num_threads=[num of the threads]
```

It configures the number of the threads within the operation. It is set to number of CPU cores by default.
 
You can find more detail in [PyTorch](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).
