#!/usr/bin/env bash

set -e
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_DIR
NUM_PROC=1
if [[ -n $(command -v nproc) ]]; then
  NUM_PROC=$(nproc)
elif [[ -n $(command -v sysctl) ]]; then
  NUM_PROC=$(sysctl -n hw.ncpu)
fi

VERSION=$1
FLAVOR=$2
if [[ ! -d libtorch_android/"$FLAVOR" ]]; then
  if [[ $FLAVOR != "armeabi-v7a" ]] && [[ $FLAVOR != "arm64-v8a" ]] && [[ $FLAVOR != "x86" ]] && [[ $FLAVOR != "x86_64" ]]; then
    echo "$FLAVOR is not supported."
    exit 1
  else
    mkdir -p libtorch_android/"$FLAVOR"
    cd libtorch_android/"$FLAVOR"
    echo "Downloading https://publish.djl.ai/pytorch/$VERSION/android_native/${FLAVOR}_native.zip"
    curl -s "https://publish.djl.ai/pytorch/$VERSION/android_native/${FLAVOR}_native.zip" | jar xv
    mv install/include include
    cd -
  fi
fi

if [[ "$VERSION" =~ ^1\.10\..*|^1\.9\..* ]]; then
  PT_OLD_VERSION=1
fi
pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake -DANDROID_ABI="$FLAVOR" -DANDROID_NATIVE_API_LEVEL=21 -DBUILD_ANDROID=ON -DPT_OLD_VERSION=${PT_OLD_VERSION} ..
cmake --build . --config Release -- -j "${NUM_PROC}"
popd
