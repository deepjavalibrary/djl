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

VERSION=$1
ARCH=$2

pushd $WORK_DIR
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggerganov/llama.cpp.git -b $VERSION
fi

if [ ! -d "build" ]; then
  mkdir build
fi
cd build

rm -rf classes
mkdir classes
javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/llama/jni/LlamaLibrary.java -h include -d classes
cmake .. -DOS_NAME=$PLATFORM -DOS_ARCH=$ARCH -DLLAMA_VERSION=$VERSION
cmake --build . --config Release -- -j "${NUM_PROC}"

popd

# for nightly ci
if [[ $PLATFORM == 'darwin' ]]; then
  mkdir -p build/jnilib/osx-$ARCH
  cp -f build/libdjl_llama.dylib build/jnilib/osx-$ARCH/
  cp -f build/llama.cpp/libllama.dylib build/jnilib/osx-$ARCH/
  if [[ $ARCH == 'aarch64' ]]; then
    cp -f llama.cpp/ggml-metal.metal build/jnilib/osx-$ARCH/
  fi
elif [[ $PLATFORM == 'linux' ]]; then
  mkdir -p build/jnilib/linux-$ARCH
  cp -f build/libdjl_llama.so build/jnilib/linux-$ARCH/
  cp -f build/llama.cpp/libllama.so build/jnilib/linux-$ARCH/
fi
