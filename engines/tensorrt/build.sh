#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR

VERSION="$(cat ../../gradle.properties | awk -F '=' '/trt_version/ {print $2}')"

if [ ! -d "trt" ];
then
  git clone --recurse-submodules https://github.com/NVIDIA/TensorRT.git -b v$VERSION trt
  cp -f trt/parsers/onnx/NvOnnxParser.h trt/include
fi

if [ ! -d "build" ];
then
  mkdir build
fi
cd build
if [ ! -d "classes" ];
then
  mkdir classes
fi
javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/tensorrt/jni/TrtLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release -- -j "$(nproc)"
