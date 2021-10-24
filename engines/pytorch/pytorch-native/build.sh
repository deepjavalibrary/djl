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
CXX11ABI="-cxx11-abi"
if [[ $3 == "precxx11" ]]; then
  CXX11ABI=""
fi

if [[ ! -d "libtorch" ]]; then
  if [[ $PLATFORM == 'linux' ]]; then
    if [[ $FLAVOR == "cpu" ]] || [[ $FLAVOR == "cu111" ]] || [[ $FLAVOR == "cu102" ]]; then
      curl -s https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip | jar xv
    else
      echo "$1 is not supported."
      exit 1
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip | jar xv
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
