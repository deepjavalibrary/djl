#!/usr/bin/env bash
./gradlew clean publish -Plocal
cp -R examples/build/repo .
cp -R api/build/repo .
cp -R model-zoo/build/repo .
cp -R repository/build/repo .
cp -R basicdataset/build/repo .
cp -R mxnet/engine/build/repo .
cp -R mxnet/mxnet-model-zoo/build/repo .
cp -R mxnet/native/build/repo .
