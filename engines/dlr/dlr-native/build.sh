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

VERSION=v"$(cat ../../../gradle.properties | awk -F '=' '/dlr_version/ {print $2}')"

if [ ! -d "neo-ai-dlr" ];
then
  git clone https://github.com/neo-ai/neo-ai-dlr.git -b $VERSION --recursive
fi

if [ ! -d "build" ];
then
  mkdir build
fi
cd build
if [ ! -d "classes" ];
then
  mkdir classes
fi
javac -sourcepath ../../dlr-engine/src/main/java/ ../../dlr-engine/src/main/java/ai/djl/dlr/jni/DlrLibrary.java -h include -d classes
cmake ..
cmake --build . --config Release -- -j "${NUM_PROC}"
