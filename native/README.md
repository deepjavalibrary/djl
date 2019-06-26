## Instruction

### Step 1: Rename
Change the version with the current date

Copy and rename the file in properties to `mxnet.properties` to `binaries/native/lib` folder

### Step 2: Copy library

Copy the library so files and all dependencies in `native/lib`

### Step 3: Publish

do the followings to publish your package:

```bash
./gradlew -Plocal -Pclassifier=osx-x86_64 -Partifact=mxnet-native-mkl publish
```

The available names:
- mxnet-native-cu100mkl
- mxnet-native-cu100
- mxnet-native-cu92mkl
- mxnet-native-cu92
- mxnet-native-cu90mkl
- mxnet-native-cu90
- mxnet-native-mkl
- mxnet-native-min
