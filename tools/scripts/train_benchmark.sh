#!/usr/bin/env bash
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-b 32 -e 10 -o logs/" &> res50Cifar10Imp.log
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-b 32 -e 10 -o logs/ -s" &> res50Cifar10Sym.log
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-b 32 -e 10 -o logs/ -s -p" &> res50Cifar10SymPretrain.log
