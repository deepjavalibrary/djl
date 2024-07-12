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

VERSION=v$1
ARCH=$2
FLAVOR=$3

pushd $WORK_DIR
if [ ! -d "tokenizers" ]; then
  git clone https://github.com/huggingface/tokenizers -b $VERSION
fi

if [ ! -d "build" ]; then
  mkdir build
fi

rm -rf build/classes
mkdir build/classes
javac -sourcepath src/main/java/ src/main/java/ai/djl/huggingface/tokenizers/jni/TokenizersLibrary.java -h build/include -d build/classes
javac -sourcepath src/main/java/ src/main/java/ai/djl/engine/rust/RustLibrary.java -h build/include -d build/classes

RUST_MANIFEST=rust/Cargo.toml
if [[ "$FLAVOR" = "cpu"* ]]; then
  cargo build --manifest-path $RUST_MANIFEST --release
elif [[ "$FLAVOR" = "cu"* && "$FLAVOR" > "cu121" ]]; then
  cargo build --manifest-path $RUST_MANIFEST --release --features cuda,cublaslt,flash-attn
else
  cargo build --manifest-path $RUST_MANIFEST --release
fi

# for nightly ci
if [[ $PLATFORM == 'darwin' ]]; then
  mkdir -p build/jnilib/osx-$ARCH/$FLAVOR
  cp -f rust/target/release/libdjl.dylib build/jnilib/osx-$ARCH/$FLAVOR/libtokenizers.dylib
elif [[ $PLATFORM == 'linux' ]]; then
  mkdir -p build/jnilib/linux-$ARCH/$FLAVOR
  cp -f rust/target/release/libdjl.so build/jnilib/linux-$ARCH/$FLAVOR/libtokenizers.so
fi
