# DJL - MXNet native library

## Publishing MXNet native library

### Step 1: Prepare MXNet native library

Extract MXNet native library files from MXNet python pip wheel or build them from source.
Make sure collect all dependencies. Use the following commands include all the libmxnet.so dependencies:
```bash
# osx
otool -L libmxnet.dylib

# ubuntu
ldd libmxnet.so

# Windows
dumpbin /dependents libmxnet.dll
```

The example dependencies list for macOS mkl is:
- libmxnet.dylib (Be sure to rename libmxnet.so to libmxnet.dylib for macOS)

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


### Step 2: Upload files to s3 bucket

The example list for s3 files is: 
- s3://djl-ai/publish/mxnet-1.6.0/linux/common/libgfortran.so.3
- s3://djl-ai/publish/mxnet-1.6.0/linux/common/libquadmath.so.0
- s3://djl-ai/publish/mxnet-1.6.0/linux/cu101mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/linux/cu92mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/linux/mkl/libmxnet.so
- s3://djl-ai/publish/mxnet-1.6.0/osx/mkl/libmxnet.dylib
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libgcc_s_seh-1.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libgfortran-3.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libopenblas.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/common/libquadmath-0.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/cu101mkl/libmxnet.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/cu92mkl/libmxnet.dll
- s3://djl-ai/publish/mxnet-1.6.0/win/mkl/libmxnet.dll

### Step 3: Test publishing

Run the following commands to prepare your package:

```bash
cd mxnet/native

# Download native files and put into right folder
./gradlew dMNL

# Publish to build/repo folder
./gradlew publish

# Publish to sonatype snapshot repo
./gradlew publish -Psnapshot
```

### Step 4: Use GitHub action to publish MXNet native library

We have weekly GitHub pipeline to publish snapshot automatically. We can also use GitHub REST API to manually trigger a publish:

```bash
# manually trigger publish a snapshot release
curl -XPOST -u "USERNAME:PERSONAL_TOKEN" -H "Accept: application/vnd.github.everest-preview+json" -H "Content-Type: application/json" https://api.github.com/repos/USERNAME/RESPOSITORY_NAME/dispatches --data '{"event_type": “mxnet-snapshot-pub"}'

# trigger publishing MXNet to sonatype stagging
curl -XPOST -u "USERNAME:PERSONAL_TOKEN" -H "Accept: application/vnd.github.everest-preview+json" -H "Content-Type: application/json" https://api.github.com/repos/USERNAME/RESPOSITORY_NAME/dispatches --data '{"event_type": “mxnet-staging-pub"}'
```
