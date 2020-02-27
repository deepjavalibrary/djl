# Step to build this project

## Prerequisite
You need to install `cmake` and C++ compiler on your machine in order to build

### Linux
```
apt install cmake g++
```

## Build

Use the following command to build pytorch JNI library:

### Mac/Linux
```
./gradlew buildJNI
```

The output file `torch_djl` will be copied to `pytorch-engine/build` folder
