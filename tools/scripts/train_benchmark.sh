#!/usr/bin/env bash
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-b 32 -e 10 -o logs/" &> res50Cifar10Imp.log
mv logs/training.log ./res50Cifar10ImpTime.log
mv logs/memory.log ./res50Cifar10ImpMemory.log
./gradlew run -Dmain=ai.djl.examples.training.transferlearning.TrainResnetWithCifar10 --args="-b 32 -e 10 -o logs/ -s true" &> res50Cifar10Sym.log
mv logs/training.log ./res50Cifar10SymTime.log
mv logs/memory.log ./res50Cifar10SymMemory.log
