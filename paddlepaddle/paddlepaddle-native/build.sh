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
      curl -s https://alpha-djl-demos.s3.amazonaws.com/temp/paddle_inference_install_dir-gcc54-2.0-openblas.tgz -o paddle.tgz
    else
      echo "$1 is not supported."
      exit 1
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    curl -s https://paddle-inference-lib.bj.bcebos.com/mac%2F2.0-rc%2Fcpu_avx_openblas%2Fpaddle_inference_install_dir.tgz -o paddle.tgz
  else
    echo "$PLATFORM is not supported."
    exit 1
  fi
  tar -xvzf paddle.tgz
  mv paddle_inference_install_dir paddle
fi

rm -rf build
mkdir build && cd build

rm -rf classes
mkdir classes
javac -sourcepath ../../paddlepaddle-engine/src/main/java/ ../../paddlepaddle-engine/src/main/java/ai/djl/paddlepaddle/jni/PaddleLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release -- -j "${NUM_PROC}"

popd
