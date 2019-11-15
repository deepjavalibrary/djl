#!/usr/bin/env bash
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c 100 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> multithread.log
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true -DMXNET_THREAD_SAFE_INFERENCE=true --args="-c 100 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> multithread_threadsafe.log
export MXNET_CPU_WORKER_NTHREADS=2
#export MXNET_OMP_NUM_THREADS=1
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark -Dcollect-memory=true --args="-c 1600 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> res18_worker.log
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true -DMXNET_THREAD_SAFE_INFERENCE=true --args="-c 100 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> multithread_threadsafe_worker.log
