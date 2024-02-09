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
    if [[ ! "$FLAVOR" =~ ^(cpu|cu102|cu113|cu116|cu117|cu118|cu121)$ ]]; then
      echo "$FLAVOR is not supported."
      exit 1
    fi

    if [[ $ARCH == 'aarch64' ]]; then
      curl -s https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch${AARCH64_CXX11ABI}-shared-with-deps-${VERSION}-aarch64.zip | jar xv >/dev/null
    else
      curl -s https://download.pytorch.org/libtorch/${FLAVOR}/libtorch${CXX11ABI}-shared-with-deps-${VERSION}%2B${FLAVOR}.zip | jar xv >/dev/null
    fi
  elif [[ $PLATFORM == 'darwin' ]]; then
    if [[ "$VERSION" =~ ^(2.2.)* ]]; then
      if [[ $ARCH == 'aarch64' ]]; then
        curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${VERSION}.zip | jar xv >/dev/null
      else
        curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-x86_64-${VERSION}.zip | jar xv >/dev/null
      fi
    else
      if [[ $ARCH == 'aarch64' ]]; then
        curl -s https://djl-ai.s3.amazonaws.com/publish/pytorch/${VERSION}/libtorch-macos-${VERSION}-aarch64.zip | jar xv >/dev/null
      else
        curl -s https://download.pytorch.org/libtorch/cpu/libtorch-macos-${VERSION}.zip | jar xv >/dev/null
      fi
    fi
  else
    echo "$PLATFORM is not supported."
    exit 1
  fi
fi

if [[ "$FLAVOR" = cu* ]]; then
  USE_CUDA=1
fi

pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=libtorch -DPT_VERSION=${PT_VERSION} -DUSE_CUDA=$USE_CUDA ..
cmake --build . --config Release -- -j "${NUM_PROC}"
if [[ "$FLAVOR" = cu* ]]; then
  # avoid link with libcudart.so.11.0
  sed -i -r "s/\/usr\/local\/cuda(.{5})?\/lib64\/lib(cudart|nvrtc).so//g" CMakeFiles/djl_torch.dir/link.txt
  rm libdjl_torch.so
  . CMakeFiles/djl_torch.dir/link.txt
fi

if [[ $PLATFORM == 'darwin' ]]; then
  install_name_tool -add_rpath @loader_path libdjl_torch.dylib
fi

popd
