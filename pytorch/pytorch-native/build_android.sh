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

VERSION=1.9.0
if [[ ! -d libtorch_android/"$1" ]]; then
    if [[ $1 != "armeabi-v7a" ]] && [[ $1 != "arm64-v8a" ]] && [[ $1 != "x86" ]] && [[ $1 != "x86_64" ]]; then
        echo "$1 is not supported."
        exit 1
    else
        mkdir -p libtorch_android/"$1"
        cd libtorch_android/"$1"
        curl -s https://publish.djl.ai/pytorch-"{$VERSION}/android_native/{$1}"_native.zip | jar xv
        mv install/include include
        cd ../../
    fi
fi

pushd .

rm -rf build
mkdir build && cd build
mkdir classes
javac -sourcepath ../../pytorch-engine/src/main/java/ ../../pytorch-engine/src/main/java/ai/djl/pytorch/jni/PyTorchLibrary.java -h include -d classes
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK"/build/cmake/android.toolchain.cmake -DANDROID_ABI="$1" -DANDROID_NATIVE_API_LEVEL=21 -DBUILD_ANDROID=ON  ..
cmake --build . --config Release -- -j "${NUM_PROC}"
popd
