#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
fi
PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

pushd $WORK_DIR

echo "Trying to find paddle folder..."

if [[ ! -d "paddle" ]]; then
  echo "Folder not found. Downloading C++ package..."
  if [[ $PLATFORM == 'linux' ]]; then
    if [[ $1 == "cpu" ]]; then
      curl -s https://alpha-djl-demos.s3.amazonaws.com/temp/paddle202/paddle_inference_install_dir-2.0.2-openblas-gcc54-ubuntu.tgz -o paddle.tgz
      tar -xvzf paddle.tgz
      mv paddle_inference_install_dir paddle
    elif [[ $1 == "cu101" ]]; then
      curl -s https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda10.1-cudnn7-avx-mkl/paddle_inference.tgz -o paddle.tgz
        tar -xvzf paddle.tgz
        mv paddle_inference paddle
    elif [[ $1 == "cu102" ]]; then
      curl -s https://paddle-inference-lib.bj.bcebos.com/2.0.2-gpu-cuda10.2-cudnn8-avx-mkl/paddle_inference.tgz -o paddle.tgz
        tar -xvzf paddle.tgz
        mv paddle_inference paddle
    else
      echo "$1 is not supported."
      exit 1
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    curl -s https://paddle-inference-lib.bj.bcebos.com/mac/2.0.2/cpu_avx_openblas/paddle_inference.tgz -o paddle.tgz
    tar -xvzf paddle.tgz
    mv paddle_inference_install_dir paddle
  else
    echo "$PLATFORM is not supported."
    exit 1
  fi

fi

rm -rf build
mkdir build && cd build

rm -rf classes
mkdir classes
javac -sourcepath ../../paddlepaddle-engine/src/main/java/ ../../paddlepaddle-engine/src/main/java/ai/djl/paddlepaddle/jni/PaddleLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release -- -j "${NUM_PROC}"

popd
