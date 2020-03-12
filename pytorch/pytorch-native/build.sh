#!/usr/bin/env bash

set -e
export WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
fi

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
cmake --build . --config Release -- -j "${NUM_PROC}"

if [[ $PLATFORM == 'darwin' ]]; then
  install_name_tool -delete_rpath "${WORK_DIR}/libtorch/lib" libdjl_torch.dylib
  install_name_tool -add_rpath  @loader_path libdjl_torch.dylib
fi

popd
