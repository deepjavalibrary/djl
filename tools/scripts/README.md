# Start your benchmark with DJL

This is the documentation to setup benchmark for DJL on Deep Learning Benchmark.

## `create_with_nvidia_docker.sh`
run the `create_with_nvidia_docker.sh` script in an machine to create your base environment.
Currently, we use a clean Ubuntu 16.04 build with this script.

## `build.gradle`
The `build.gradle` file is used to run inside the docker/ami. It can be used to run all examples available in DJL.

## `publish.sh`
The file used to collect all artifacts and update the existing repo.
It can be done in the following way. In root directory
```
bash tools/scripts/publish.sh
aws s3 sync repo/ s3://djl-ai/benchmark-repo --acl public-read
```

## `benchmark.sh`
This file can be used to run the benchmark on all examples.
