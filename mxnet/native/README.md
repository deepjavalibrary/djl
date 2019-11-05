## Instruction

### Step 1: Copy library

Copy the library files and all dependencies in `<path_to_your_DJL>/mxnet/native/src/main/resources/binaries/<osName>/native/lib`.

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

The valid OS names are:
- `osx`
- `linux`
- `win`

### Step 2: Publish

do the followings to prepare your package:

```bash
cd mxnet/native
./gradlew -Pflavor=mkl publish
```

The available flavor names are:
- cu101mkl
- cu101
- cu92mkl
- cu92
- cu90mkl
- cu90
- mkl
- min

Supported classifier names are:
- osx-x86_64
- win-x86_64
- linux-x86_64
- linux-arm
