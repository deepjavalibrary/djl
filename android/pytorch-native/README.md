
# Pytorch Android Build Procedure

## Prerequisites

```
# install needed python library
pip install pyyaml

# Don't run the following command if you don't have python3 installed
export PYTHON=python3
```

## Linux (Ubuntu 20.04) android-sdk install

```
sudo apt-get install android-sdk cmake
export ANDROID_SDK_ROOT=/usr/lib/android-sdk
sudo chown -R ubuntu:ubuntu $ANDROID_SDK_ROOT
```

## Mac android-sdk install

```
brew install android-sdk cmake
export ANDROID_SDK_ROOT=/Users/$USER/Library/Android/sdk
sudo chown -R $USER $ANDROID_SDK_ROOT 
```

<br>
<br>
<br>
<br>
<br>
<br>

## Install Command line only tools

Find latest command line only tools: [https://developer.android.com/studio#downloads](https://developer.android.com/studio#downloads:~:text=Command%20line%20tools%20only)

```
mkdir -p $ANDROID_SDK_ROOT/cmdline-tools
cd $ANDROID_SDK_ROOT/cmdline-tools

# From link above, set ANDROID_CLT equal to the name of Linux's latest SDK tools package release
export ANDROID_CLT=commandlinetools-linux-8512546_latest.zip

curl -O https://dl.google.com/android/repository/${ANDROID_CLT}
unzip $ANDROID_CLT
rm $ANDROID_CLT

mv cmdline-tools tools
```

## Install Android NDK

See GitHub actions to ensure latest NDK_VERSION: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml)

```
export NDK_VERSION=21.0.6113669
echo "y" | sudo ${ANDROID_SDK_ROOT}/cmdline-tools/tools/bin/sdkmanager --install "ndk;${NDK_VERSION}"
```

## Build PyTorch Android native binary from source

See: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml)

Prerequisites:
* If you do not have `cmake` installed, run `brew install cmake`
* If you do not have the python library `pyyaml` installed, run `pip install pyyaml`
* If you have `python3` installed, run `export PYTHON=python3`

```
# cd into whatever directory holds your djl directory
export PYTORCH_VERSION=1.11.0
export ANDROID_NDK=${ANDROID_SDK_ROOT}/ndk/${NDK_VERSION}

git clone https://github.com/pytorch/pytorch -b "v${PYTORCH_VERSION}" --recursive
cd pytorch

# need to build all four ABIs
export ANDROID_ABI="armeabi-v7a with NEON"
#export ANDROID_ABI=arm64-v8a
#export ANDROID_ABI=x86
#export ANDROID_ABI=x86_64
BUILD_LITE_INTERPRETER=0 ./scripts/build_android.sh 

cd build_android
zip -r ~/${ANDROID_BI}_native.zip install/include lib
cd ..
# clean up pytorch repository, before build next ABI
git clean -dffx .
```

The zip files will be uploaded to S3 bucket in CI build

## Build PyTorch Android JNI

See: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_jni_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_jni_s3_pytorch_android.yml)

```
export ANDROID_NDK=${ANDROID_SDK_ROOT}/ndk/${NDK_VERSION}

cd ../djl/engines/pytorch/pytorch-native

# to avoid download pytorch native from S3, manually unzip pytoch native
mkdir libtorch_android
cd libtorch_android
mkdir arm64-v8a armeabi-v7a x86 x86_64
cd arm64-v8a && unzip ~/arm64-v8a_native.zip && mv install/include . && cd ..
cd armeabi-v7a && unzip ~/armeabi-v7a_native.zip && mv install/include . && cd ..
cd x86 && unzip ~/x86_native.zip && mv install/include . && cd ..
cd x86_64 && unzip ~/x86_64_native.zip && mv install/include . && cd ..

cd ..

./gradlew compileAndroidJNI -Ppt_version=${PYTORCH_VERSION}
```

`jnilib/0.18.0/android` folder will be created after build, and shared library will be uploaded to S3 in CI build

## Build PyTorch android library (.aar) and publish to Sonatype snapshot repo

See: [ https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/publish_android_packages.yml]( https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/publish_android_packages.yml)

```
# move into djl/android directory
cd ../../../android 

# To avoid download jni from S3, manually copy them
mkdir -p pytorch-native/jnilib
cp -r ../engines/pytorch/pytorch-native/jnilib/0.18.0/android/* pytorch-native/jnilib

./gradlew :pytorch-native:assemble
# publish to local maven repo (~/.m2 folder)
./gradlew pTML
```

`ai.djl.android:pytorch-native:0.18.0-SNAPSHOT`  will be published to mavenLocal (your local ~/.m2 folder)

## Build PyTorch Android demo projects

See: [https://github.com/deepjavalibrary/djl-demo/tree/master/android/pytorch_android](https://github.com/deepjavalibrary/djl-demo/tree/master/android/pytorch_android)

From Android Studio, with an emulator turned on, run the following commands

```
cd djl-demo/android/pytorch_android/style_transfer_cyclegan
./gradlew build
./gradlew iD
```
