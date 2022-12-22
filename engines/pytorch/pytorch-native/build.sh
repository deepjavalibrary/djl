#!/usr/bin/env bash

set -ex
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
  NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
  NUM_PROC=$(sysctl -n hw.ncpu)
fi

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
VERSION=$1
FLAVOR=$2
AARCH64_CXX11ABI="-cxx11"
CXX11ABI="-cxx11-abi"
if [[ $3 == "precxx11" ]]; then
  CXX11ABI=""
  AARCH64_CXX11ABI=""
fi
ARCH=$4

if [[ ! -d "libtorch" ]]; then
  if [[ $PLATFORM == 'linux' ]]; then
    if [[ ! "$FLAVOR" =~ ^(cpu|cu102|cu113|cu116|cu117)$ ]]; then
      echo "$FLAVOR is not supported."
      exit 1
    fi

    if [[ $ARCH == 'aarch64' ]]; then
      curl -s https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch${AARCH64_CXX11ABI}-shared-with-deps-${VERSION}-aarch64.zip | jar xv
    else
      curl -s https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip | jar xv
    fi

  elif [[ $PLATFORM == 'darwin' ]]; then
    if [[ $ARCH == 'aarch64' ]]; then
      curl -s https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-macos-${VERSION}-aarch64.zip | jar xv
    else
      curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip | jar xv
    fi
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
  install_name_tool -add_rpath @loader_path libdjl_torch.dylib
fi

popd
