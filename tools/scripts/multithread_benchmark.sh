#!/usr/bin/env bash
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg
export export MXNET_ENGINE_TYPE=NaiveEngine
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark -Dcollect-memory=true --args="-c 2000 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> res18.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark -Dcollect-memory=true --args="-c 1600 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> res18.log
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c 100 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> multithread.log
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true -DMXNET_THREAD_SAFE_INFERENCE=true --args="-c 100 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> multithread_threadsafe.log
./gradlew run -Dmain=ai.djl.examples.inference.MultithreadedBenchmark -Dcollect-memory=true --args="-c 100 -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'}" &> multithread_imp.log

