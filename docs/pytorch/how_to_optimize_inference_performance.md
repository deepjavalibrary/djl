## Optimize the PyTorch inference performance

### Multithreading Inference
To use multithreading inference feature, we have to disable GC to close the NDArray by
```
-Dai.djl.pytorch.disable_close_resource_on_finalize=true
```
Please make sure all the NDArrays are attached to the NDManager.
It is expected to be fixed in the future.

###
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
