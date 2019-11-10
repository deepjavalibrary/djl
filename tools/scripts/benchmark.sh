#!/usr/bin/env bash
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/docs/images/3dogs.jpg
curl -O https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/pose/soccer.png
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i 3dogs.jpg -r {'layers':'18','flavor':'v1'}" &> res18.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i 3dogs.jpg -r {'layers':'50','flavor':'v2'}" &> res50.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i 3dogs.jpg -r {'layers':'152','flavor':'v1d'}" &> res152.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i 3dogs.jpg -r {'layers':'50','flavor':'v1','dataset':'cifar10'}" &> res50Cifar10.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i 3dogs.jpg -m -r {'layers':'50','flavor':'v1'}" &> res50Cifar10Imp.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i soccer.png -n SSD -r {'size':'512','backbone':'resnet50'}" &> ssdresnet50.log
./gradlew run -Dmain=ai.djl.examples.inference.Benchmark --args="-c 1000 -i soccer.png -n SSD -r {'size':'512','backbone':'vgg16'}" &> ssdvgg16.log
