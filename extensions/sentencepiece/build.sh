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

VERSION=v0.1.95

pushd $WORK_DIR
if [ ! -d "sentencepiece" ];
then
  git clone https://github.com/google/sentencepiece.git -b $VERSION
fi

if [ ! -d "build" ];
then
  mkdir build
fi
cd build

rm -rf classes
mkdir classes
javac -sourcepath ../src/main/java/ ../src/main/java/ai/djl/sentencepiece/jni/SentencePieceLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release -- -j "${NUM_PROC}"

popd

# for nightly ci
if [[ $PLATFORM == 'darwin' ]]; then
  mkdir -p build/jnilib/osx-x86_64
  cp -f build/libsentencepiece_native.dylib build/jnilib/osx-x86_64/
elif [[ $PLATFORM == 'linux' ]]; then
  mkdir -p build/jnilib/linux-x86_64
  cp -f build/libsentencepiece_native.so build/jnilib/linux-x86_64/
fi
