#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

VERSION=v$1
ARCH=$2
FLAVOR=$3

pushd "$WORK_DIR"
if [ ! -d "tokenizers" ]; then
  git clone https://github.com/huggingface/tokenizers -b "$VERSION"
fi

if [ ! -d "build" ]; then
  mkdir build
fi

rm -rf build/classes
mkdir build/classes
javac -sourcepath src/main/java/ src/main/java/ai/djl/huggingface/tokenizers/jni/TokenizersLibrary.java -h build/include -d build/classes
javac -sourcepath src/main/java/ src/main/java/ai/djl/engine/rust/RustLibrary.java -h build/include -d build/classes

function copy_files() {
  # for nightly ci
  arch="$1"
  flavor="$2"
  if [[ $PLATFORM == 'darwin' ]]; then
    mkdir -p "build/jnilib/osx-$arch/$flavor"
    cp -f rust/target/release/libdjl.dylib "build/jnilib/osx-$arch/$flavor/libtokenizers.dylib"
  elif [[ $PLATFORM == 'linux' ]]; then
    mkdir -p "build/jnilib/linux-$arch/$flavor"
    cp -f rust/target/release/libdjl.so "build/jnilib/linux-$arch/$flavor/libtokenizers.so"
  fi
}

RUST_MANIFEST=rust/Cargo.toml
if [[ "$FLAVOR" = "cpu"* ]]; then
  cargo build --manifest-path $RUST_MANIFEST --release
  copy_files "$ARCH" "$FLAVOR"
elif [[ "$FLAVOR" = "cu"* && "$FLAVOR" > "cu121" ]]; then
  CUDA_COMPUTE_CAP=80 cargo build --manifest-path $RUST_MANIFEST --release --features cuda,flash-attn
  copy_files "$ARCH" "${FLAVOR}-80"

  cargo clean --manifest-path $RUST_MANIFEST
  CUDA_COMPUTE_CAP=86 cargo build --manifest-path $RUST_MANIFEST --release --features cuda,flash-attn
  copy_files "$ARCH" "${FLAVOR}-86"

  cargo clean --manifest-path $RUST_MANIFEST
  CUDA_COMPUTE_CAP=89 cargo build --manifest-path $RUST_MANIFEST --release --features cuda,flash-attn
  copy_files "$ARCH" "${FLAVOR}-89"

  cargo clean --manifest-path $RUST_MANIFEST
  CUDA_COMPUTE_CAP=90 cargo build --manifest-path $RUST_MANIFEST --release --features cuda,flash-attn
  copy_files "$ARCH" "${FLAVOR}-90"
else
  cargo build --manifest-path $RUST_MANIFEST --release
  copy_files "$ARCH" "$FLAVOR"
fi
