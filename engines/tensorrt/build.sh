#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR

VERSION="$(awk -F '=' '/tensorrt/ {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)"

if [ ! -d "trt" ];
then
  git clone https://github.com/NVIDIA/TensorRT.git -b $VERSION trt
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
