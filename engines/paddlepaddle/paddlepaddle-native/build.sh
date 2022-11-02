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

# https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html 2.3.2

if [[ ! -d "paddle" ]]; then
  echo "Folder not found. Downloading C++ package..."
  if [[ $PLATFORM == 'linux' ]]; then
    if [[ $1 == "cpu" ]]; then
      curl -s https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz -o paddle.tgz
      tar -xvzf paddle.tgz
      mv paddle_inference paddle
    elif [[ $1 == "cu102" ]]; then
      curl -s https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda10.2_cudnn7.6.5_trt6.0.1.5/paddle_inference.tgz -o paddle.tgz
        tar -xvzf paddle.tgz
        mv paddle_inference paddle
    elif [[ $1 == "cu112" ]]; then
      curl -s https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Linux/GPU/x86-64_gcc5.4_avx_mkl_cuda11.2_cudnn8.2.1_trt8.0.3.4/paddle_inference.tgz -o paddle.tgz
        tar -xvzf paddle.tgz
        mv paddle_inference paddle
    else
      echo "$1 is not supported."
      exit 1
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    curl -s https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/MacOS/CPU/x86-64_clang_avx_openblas/paddle_inference_install_dir.tgz -o paddle.tgz
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
