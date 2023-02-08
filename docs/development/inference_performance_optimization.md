# Inference Performance Optimization

The document covers several tricks of how you can tune your inference performance based on the engine you use 
including multithreading support, engine threads configuration and how to enable DNNL(MKLDNN).

## Multithreading Support

One of the advantage of Deep Java Library (DJL) is Multi-threaded inference support.
It can help to increase the throughput of your inference on multi-core CPUs and GPUs and reduce
memory consumption compare to Python.

DJL `Predictor` is not designed to be thread-safe (although some implementation is),
we recommend creating a new [Predictor](https://javadoc.io/doc/ai.djl/api/latest/ai/djl/inference/Predictor.html) for each thread.

For a reference implementation, see [Multi-threaded Benchmark](https://github.com/deepjavalibrary/djl-serving/blob/master/benchmark/src/main/java/ai/djl/benchmark/MultithreadedBenchmark.java).

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

Note that MxNet uses thread_local storage: Every thread that performs inference will allocate memory. In order to avoid memory leaks it is necessary to call MxNet from a fixed size thread pool. See https://github.com/apache/incubator-mxnet/issues/16431#issuecomment-562052116.

### PyTorch

#### graph optimizer

PyTorch graph executor optimizer (JIT tensorexpr fuser) is enabled by default. This may impact
the inference latency for a few inference calls. You can disable graph executor optimizer globally
by setting the following system properties:

```
-Dai.djl.pytorch.graph_optimizer=false
```

The graph executor optimizer is per thread configuration. If you want to disable it in a per model
basis, you have to call the following method in each inference thread:

```java
JniUtils.setGraphExecutorOptimize(false);
```

#### oneDNN(MKLDNN) acceleration
Unlike TensorFlow and Apache MXNet, PyTorch by default doesn't enable MKLDNN which is treated as a device type like CPU and GPU.
You can enable it by

```
-Dai.djl.pytorch.use_mkldnn=true
```

You might see the exception if certain data type or operator is not supported with the oneDNN device.

#### CuDNN acceleration
PyTorch has a special flags that used for CNN or related network speed up. If your input size won't change frequently,
you may benefit from enabling this configuration in your model:

```
-Dai.djl.pytorch.cudnn_benchmark=true
```

If your input shape changed frequently, this change may stall your performance. For more information, check this 
[article](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner).

#### Thread configuration
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

#### GPU (Disabling Graph Executor Optimization)

When the first inference is made on a new batch size, torchscript generates an optimized execution graph for the model. 
This optimization can take some time and is done for each batch size you use. To disable this feature, you can use the following code:

```
Engine.getEngine("PyTorch"); // Make sure PyTorch engine is loaded
JniUtils.setGraphExecutorOptimize(false);
```

By default, Graph Executor Optimization is on. 
It is important to note that it must be disabled for each thread that you are using the model from.
If you forget to disable it on a thread, it will cause the optimization to be performed and delay any running/pending executions.

Disabling Graph Executor Optimization causes a maximum throughput and performance loss that can depend on the model and hardware.
This should only be disabled when you do not have the time to "warmup" a model with your given batch sizes, or you want to use dynamic batch sizes. 

### TensorFlow

#### Multithreading Inference
You can follow the same steps as other engines for running multithreading inference using TensorFlow engine.
It's recommended to use one `Predictor` for each thread and avoid using a new `Predictor` for each inference call.
You can refer to our [Multithreading Benchmark](https://github.com/deepjavalibrary/djl-serving/blob/master/benchmark/src/main/java/ai/djl/benchmark/MultithreadedBenchmark.java) as an example,
here is how to run it using TensorFlow engine.

```bash
cd djl-serving
./gradlew benchmark --args='-e TensorFlow -c 100 -t -1 -u djl://ai.djl.tensorflow/resnet/0.0.1/resnet50 -s 1,224,224,3'
```

#### oneDNN(MKLDNN) acceleration
By default, TensorFlow engine comes with oneDNN enabled, no special configuration needed.

#### Thread configuration
It's recommended to use 1 thread for operator parallelism during multithreading inference. 
You can configure it by setting the following 3 envinronment variables:

```bash
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
```

### ONNXRuntime

#### Thread configuration

You can use the following settings for thread optimization in Criteria

```
.optOption("interOpNumThreads", <num_of_thread>)
.optOption("intraOpNumThreads", <num_of_thread>)
```

Tips: Set to 1 on both of them at the beginning to see the performance. 
Then set to total_cores/total_java_inference_thread on one of them to see how performance goes.

#### (GPU) TensorRT Backend

If you have tensorRT installed, you can try with the following backend on ONNXRuntime for performance optimization in Criteria

```
.optOption("ortDevice", "TensorRT")
```

