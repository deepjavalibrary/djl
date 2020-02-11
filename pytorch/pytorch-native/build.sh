#!/usr/bin/env bash

set -e

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
if [[ ! -d "libtorch" ]]; then
  if [[ $PLATFORM == 'linux' ]]; then
    curl -s https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip | jar xv
  elif [[ $PLATFORM == 'darwin' ]]; then
    curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.4.0.zip | jar xv
  else
    echo "$PLATFORM is not supported."
    exit 1
  fi
fi

pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch ..
cmake --build . --config Release

popd
