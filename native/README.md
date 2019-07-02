## Instruction

### Step 1: Copy library

Copy the library files and all dependencies in `<osName>/native/lib`. The valid OS names are:
- `osx`
- `linux`

### Step 2: Publish

do the followings to prepare your package:

```bash
./gradlew -Plocal -Pflavor=mkl publish
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
- win32-x86_64
- linux-x86_64
- linux-arm


### Step 3: Refresh repo

You can refresh the bucket by running
```bash
aws s3 sync repo/ s3://joule/repo --acl public-read
```
