# Inference Performance Optimization

This document covers several tricks of how you can tune your inference performance based on the engine you use 
including multithreading support, engine threads configuration, and how to enable DNNL(MKLDNN).

## Multithreading Support

One of the advantage of Deep Java Library (DJL) is Multi-threaded inference support.
It can help to increase the throughput of your inference on multi-core CPUs and GPUs and reduce
memory consumption compared to Python.

The DJL `Predictor` is not designed to be thread-safe.
Because some engines are not thread-safe, we use the Predictor to help cover the differences.
For engines that are thread-safe, we do nothing and for ones that are not, we will make copies as necessary.

Therefore, we recommend creating a new [Predictor](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/Predictor.html) for each thread.
That Predictor should be reused if the thread does multiple predictions.
Alternatively, you can also use a pool of Predictors or you can leverage the [DJL Serving WorkLoadManager](https://docs.djl.ai/master/docs/serving/wlm/index.html).

For a reference implementation, see [Multi-threaded Benchmark](https://github.com/deepjavalibrary/djl-serving/blob/master/benchmark/src/main/java/ai/djl/benchmark/MultithreadedBenchmark.java).

In addition, you may need to set engine-specific configurations as well.
Engine-specific details are given below.
You can also reference the [list of all DJL system configurations](https://docs.djl.ai/master/docs/serving/serving/docs/configurations.html).

### Apache MXNet

#### Engine configuration
To use Apache MXNet Engine to run multi-threading, complete the following steps:

#### Enable NaiveEngine with Apache MXNet
By default, MXNet tries to execute operators lazily and with internal multi-threading designed to occupy the entire machine.
This works well during Training and if you want to focus on latency over throughput.
If you are using the MXNet engine for a Java multi-threaded inference case, you need to specify the 
'MXNET_ENGINE_TYPE' environment variable using the following command:

```bash
export MXNET_ENGINE_TYPE=NaiveEngine
```

To get the best throughput, you may also need to set the 'OMP_NUM_THREADS' environment variable:

```bash
export OMP_NUM_THREADS=1
```

### PyTorch

#### Graph Executor Optimization

The PyTorch graph executor optimizer (JIT tensorexpr fuser) is enabled by default. When the first
a few inferences is made on a **new batch size**, torchscript generates an optimized execution graph for
the model.

Disabling Graph Executor Optimization causes a maximum throughput and performance loss that
can depend on the model and hardware. Use [djl-bench](https://github.com/deepjavalibrary/djl-serving/tree/master/benchmark)
to check the impact of this optimization on your hardware.
This should only be disabled when you do not have the time to "warmup" a model with your
given batch sizes, or you want to use dynamic batch sizes.

You can disable graph executor optimizer globally by setting the following system properties:

```
-Dai.djl.pytorch.graph_optimizer=false
```

The graph executor optimizer is a per thread configuration. It is important to note that it must
be disabled for each thread that you are using the model from.
If you forget to disable it on a thread, it will cause the optimization to be performed and delay
any running/pending executions.

```java
Engine.getEngine("PyTorch"); // Make sure PyTorch engine is loaded
JniUtils.setGraphExecutorOptimize(false);
```

#### oneDNN(MKLDNN) acceleration
Unlike TensorFlow and Apache MXNet, PyTorch by default doesn't enable MKLDNN by default.
Instead, it is treated as a device type like CPU and GPU.
You can enable it by setting the environment variable:

```
-Dai.djl.pytorch.use_mkldnn=true
```

You might see an exception if a data type or operator is not supported with the oneDNN device.

#### oneDNN(MKLDNN) tuning on AWS Graviton3
AWS Graviton3(E) (e.g. c7g/m7g/r7g, c7gn and Hpc7g instances) supports BF16 format for ML acceleration. This can be enabled in oneDNN by setting the below environment variable
```
grep -q bf16 /proc/cpuinfo && export DNNL_DEFAULT_FPMATH_MODE=BF16
```
To avoid redundant primitive creation latency overhead, enable primitive caching by setting the LRU cache capacity. Please note this caching feature increases the memory footprint. It is recommended to tune the capacity to an optimal value for a given use case.

```
export LRU_CACHE_CAPACITY=1024
```

In addition to avoiding the redundant allocations, tensor memory allocation latencies can be optimized  with Linux transparent huge pages (THP). To enable THP allocations, set the following torch environment variable.
```
export THP_MEM_ALLOC_ENABLE=1
```
Please refer to [PyTorch Graviton tutorial](https://pytorch.org/tutorials/recipes/inference_tuning_on_aws_graviton.html) for more details on how to achieve the best PyTorch inference performance on AWS Graviton3 instances.

#### CuDNN acceleration
PyTorch has a special flag that is used for a CNN or related network speed up. If your input size won't change frequently,
you may benefit from enabling this configuration in your model:

```
-Dai.djl.pytorch.cudnn_benchmark=true
```

If your input shape changes frequently, this change may stall your performance. For more information, check this 
[article](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner).

#### Thread configuration
There are two configurations you can set to optimize the inference performance.

```
-Dai.djl.pytorch.num_interop_threads=[num of the interop threads]
```

This configures the number of operations the JIT interpreter forks will execute in parallel.

```
-Dai.djl.pytorch.num_threads=[num of the threads]
```

This configures the number of threads within the operation. It is set to the number of CPU cores by default.
 
You can find more details in [PyTorch](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).

### TensorFlow

#### Multithreading Inference
You can follow the same steps as for other engines for running multithreading inference using the TensorFlow engine.
It's recommended to use one `Predictor` for each thread and avoid using a new `Predictor` for each inference call.
You can refer to our [Multithreading Benchmark](https://github.com/deepjavalibrary/djl-serving/blob/master/benchmark/src/main/java/ai/djl/benchmark/MultithreadedBenchmark.java) as an example,
Here is how to run it using TensorFlow engine.

```bash
cd djl-serving
./gradlew benchmark --args='-e TensorFlow -c 100 -t -1 -u djl://ai.djl.tensorflow/resnet/0.0.1/resnet50 -s 1,224,224,3'
```

#### oneDNN(MKLDNN) acceleration
By default, the TensorFlow engine comes with oneDNN enabled, no special configuration needed.

#### Thread configuration
It's recommended to use 1 thread for operator parallelism during multithreading inference. 
You can configure it by setting the following 3 environment variables:

```bash
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
```

### ONNXRuntime

#### Thread configuration

You can use the following settings for thread optimization in Criteria

```java
Criteria.builder()
    .optOption("interOpNumThreads", <num_of_thread>)
    .optOption("intraOpNumThreads", <num_of_thread>)
    ...
```

Tips: Set to 1 on both of them at the beginning to see the performance. 
Then, set to `total_cores`/`total_java_inference_thread` on one of them to see how performance goes.

#### (GPU) TensorRT Backend

If you have tensorRT installed, you can try with the following backend on ONNXRuntime for performance optimization in Criteria

```java
Criteria.builder()
    .optOption("ortDevice", "TensorRT")
    ...
```

