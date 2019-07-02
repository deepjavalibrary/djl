## Instruction

### Step 1: Rename
Change the version with the current date

Copy and rename the file in properties to `mxnet.properties` to `binaries/native/lib` folder

### Step 2: Copy library

Copy the library so files and all dependencies in `native/lib`

### Step 3: Publish

do the followings to publish your package:

```bash
./gradlew -Plocal -Pclassifier=osx-x86_64 -Pflavor=mkl publish
```

The available flavor names are:
- cu100mkl
- cu100
- cu92mkl
- cu92
- cu90mkl
- cu90
- mkl
- min

Supported classifier names are:
- osx-x86_64
- win32-x86_64
- linux-x86_64
- linux-arm

