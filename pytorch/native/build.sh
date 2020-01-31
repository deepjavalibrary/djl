#!/bin/bash

set -e

rm -rf build
mkdir build && cd build
javac -cp ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include
cmake -DCMAKE_PREFIX_PATH=libtorch ..
cmake --build . --config Release
