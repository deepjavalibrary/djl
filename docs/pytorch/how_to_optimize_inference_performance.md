## Optimize the PyTorch inference performance

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
