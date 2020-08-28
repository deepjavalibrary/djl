# DJL - Apache MXNet native library

This module contains DJL released Apache MXNet binary files.
The source of binary can be traced from the following two ways:

- The binaries are built from source from [Apache MXNet](https://github.com/apache/incubator-mxnet) without modification.
- The binaries are obtained from the Apache MXNet python pip wheel.

## Publishing the Apache MXNet native library

### Step 1: Prepare the MXNet native library

Extract the Apache MXNet native library files from the Apache MXNet python pip wheel or build them from source.
Make sure to collect all the dependencies. Use the following commands to include all the libmxnet.so dependencies:

```bash
# osx
otool -L libmxnet.dylib

# ubuntu
ldd libmxnet.so

# Windows
dumpbin /dependents libmxnet.dll
```

The example dependencies list for macOS mkl is:

- libmxnet.dylib (be sure to rename libmxnet.so to libmxnet.dylib for macOS)

The example dependencies list for Windows mkl is:

- libmxnet.dll
- libopenblas.dll
- libgfortran-3.dll
- libquadmath-0.dll
- libgcc_s_seh-1.dll

The example dependencies list for Linux mkl is:

- libquadmath.so.0
- libgfortran.so.3
- libmxnet.so


### Step 2: Upload files to the s3 bucket

The example list of the s3 files is:

- s3://djl-ai/publish/mxnet-1.6.0/linux/common/libgfortran.so.3
- s3://djl-ai/publish/mxnet-1.6.0/linux/common/libquadmath.so.0
- s3://djl-ai/publish/mxnet-1.6.0/linux/cu102mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/linux/cu101mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/linux/mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/osx/mkl/libmxnet.dylib
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libgcc_s_seh-1.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libgfortran-3.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libopenblas.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libquadmath-0.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/cu102mkl/libmxnet.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/cu101mkl/libmxnet.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/mkl/libmxnet.dll

### Step 3: Test publishing

Run the following commands to prepare your package:

```bash
cd mxnet/native

# Download the native files and put them into the right folder
./gradlew dMNL

# Change the mxnet-native version in gradle.properties

# There are three ways to build the new mxnet-native package for testing

1. Publish to local build/repo folder
./gradlew publish

2. Publish to staging and use the new URL to test it
./gradlew publish -Pstaging

3. Publish to staging via the Github action
# Run the workflow from the web portal

# Test with the SSD Apache MXNet model
./gradlew :example:run

# After testing all three platforms(osx, linux, win), you can publish the package through sonatype.
```

### Optional: Use GitHub actions to publish the Apache MXNet native library

We have a weekly GitHub pipeline that publishes the snapshots automatically. The pipeline can also be manually triggered using the GitHub Actions web portal
