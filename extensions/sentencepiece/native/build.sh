#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
fi

VERSION=v0.1.92

pushd $WORK_DIR
if [ ! -d "sentencepiece" ];
then
  git clone https://github.com/google/sentencepiece.git
  cd sentencepiece
  mkdir build
  cd build
  cmake ..
  cmake --build . --config Release -- -j "${NUM_PROC}"
fi
popd

pushd $WORK_DIR
rm -rf build
mkdir -p build/sentencepiece/binary
cp sentencepiece/build/src/libsentencepiece.a build/sentencepiece/binary
cp -R sentencepiece/src build/sentencepiece/src
cd build
mkdir classes
javac -sourcepath ../../src/main/java/ ../../src/main/java/ai/djl/sentencepiece/jni/SentencePieceLibrary.java -h include -d classes
cmake -DCMAKE_PREFIX_PATH=sentencepiece/build ..
cmake --build . --config Release -- -j "${NUM_PROC}"

popd
