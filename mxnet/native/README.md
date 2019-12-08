# DJL - MXNet native library

## Building the library

### Step 1: Copy the library

Copy the library files and all dependencies to `<path_to_your_DJL>/mxnet/native/src/main/resources/binaries/<flavor>/<osName>/native/lib`.

Use the following commands include all the libmxnet.so dependencies:
```bash
# osx
otool -L libmxnet.dylib
# ubuntu
ldd libmxnet.so
```

The example dependencies list for macOS mkl is:
- binaries/mkl/osx/native/lib/libmxnet.dylib (Be sure to rename libmxnet.so to libmxnet.dylib for macOS)

The example dependencies list for Windows mkl is:
- binaries/mkl/win/native/lib/mxnet.dll (Be sure to rename libmxnet.dll to mxnet.dll for Windows)
- binaries/mkl/win/native/lib/libopenblas.dll
- binaries/mkl/win/native/lib/libgfortran-3.dll
- binaries/mkl/win/native/lib/libquadmath-0.dll
- binaries/mkl/win/native/lib/libgcc_s_seh-1.dll

The example dependencies list for Linux mkl is:
- binaries/mkl/linux/native/lib/libquadmath.so.0
- binaries/mkl/linux/native/lib/libgfortran.so.3
- binaries/mkl/linux/native/lib/libmxnet.so

The available MXNet native versions are:
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

Run the following commands to prepare your package:

```bash
cd mxnet/native

# Publish to build/repo folder
./gradlew publish

# If the artifact is large, increase the socket timeout
./gradlew publish -Dorg.gradle.internal.http.socketTimeout=60000 -Dorg.gradle.internal.http.connectionTimeout=60000
```
