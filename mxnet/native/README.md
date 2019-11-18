# DJL - MXNet native library

## Instruction

### Step 1: Copy library

Copy the library files and all dependencies in `<path_to_your_DJL>/mxnet/native/src/main/resources/binaries/<flavor>/<osName>/native/lib`.

Make sure to upload all the libmxnet.so dependencies by command
```bash
# osx
otool -L libmxnet.dylib
# ubuntu
ldd libmxnet.so
```

The example dependencies list for osx are
- libiomp5.dylib
- libmkldnn.0.dylib
- libmklml.dylib
- libmxnet.dylib(please make sure rename libmxnet.so to libmxnet.dylib for osx)

The available flavor names are:
- cu101mkl
- cu101
- cu92mkl
- cu92
- cu90mkl
- cu90
- mkl
- min

The valid OS names are:
- `osx`
- `linux`
- `win`

### Step 2: Publish

do the followings to prepare your package:

```bash
cd mxnet/native

# Publish to build/repo folder
./gradlew publish

# Publish to DJL s3 bucket
./gradlew publish -Ps3
```
