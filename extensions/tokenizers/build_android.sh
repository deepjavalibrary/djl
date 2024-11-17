#!/usr/bin/env bash

set -ex

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR

FLAVOR=$1

if [ ! -d "build" ]; then
  mkdir build
fi

pushd .

if [ ! -d "build" ]; then
  mkdir build
fi

rm -rf build/classes
mkdir build/classes

javac -sourcepath src/main/java/ src/main/java/ai/djl/huggingface/tokenizers/jni/TokenizersLibrary.java -h build/include -d build/classes
javac -sourcepath src/main/java/ src/main/java/ai/djl/engine/rust/RustLibrary.java -h build/include -d build/classes

cd rust/
cargo ndk -t $FLAVOR -o $WORK_DIR/build/jnilib --platform=21 build --release
cd ..
popd
