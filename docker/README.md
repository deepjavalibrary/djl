# Docker Resources
DJL provides docker files that you can use to setup containers with the appropriate environment for certain platforms.

We recommend setting up a docker container with the provided Dockerfile when developing for the following
platforms and/or engines.

## Windows
You can use the [docker file](https://github.com/deepjavalibrary/djl/blob/master/docker/windows/Dockerfile) provided by us.
Please note that this docker will only work with Windows server 2019 by default. If you want it to work with other
versions of Windows, you need to pass the version as an argument as follows:

```bash
docker build --build-arg version=<YOUR_VERSION>
```

## TensorRT
You can use the [docker file](https://github.com/deepjavalibrary/djl/blob/master/docker/tensorrt/Dockerfile) provided by us.
This docker file is a modification of the one provided by NVIDIA in
[TensorRT](https://github.com/NVIDIA/TensorRT/blob/8.4.1/docker/ubuntu-18.04.Dockerfile) to include JDK11. 
By default this sets up a container using Ubuntu 18.04 and CUDA 11.6.2. You can build the container with other versions as follows, 
but keep in mind the TensorRT software requirements outlined [here](https://github.com/NVIDIA/TensorRT#prerequisites):

```bash
docker build --build-arg OS_VERSION=<YOUR_VERSION> --build-arg CUDA_VERSION=<YOUR_VERSION>
```

To run the container, we recommend using `nvidia-docker run ...` to ensure cuda driver and runtime are compatible. 

We recommend that you follow the setup steps in the [TensorRT guide](https://github.com/NVIDIA/TensorRT) if you 
need access to the full suite of tools TensorRT provides, such as `trtexec` which can convert onnx models to 
uff tensorrt models. When following that guide, make sure to use the DJL provided 
[docker file](https://github.com/deepjavalibrary/djl/blob/master/docker/tensorrt/Dockerfile) to enable JDK11 in the docker container.
