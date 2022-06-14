
# Pytorch Android Build Procedure

Follow this setup guide in order to run DJL apps on an Android. In order to successfully run the apps, you must install several Android items and build Android ABIs outlined below.

## Prerequisites

```
# Run the following command if you have python3 installed
export PYTHON=python3

# install needed python libraries
pip3 install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions
```
## Only run the android-sdk install specific to your OS, then proceed to installing command line tools below

This will install the android-sdk on your machine as well as python3. It sets the ANDROID_HOME environment variable for use throughout this setup guide, and gives you ownership to the ANDROID_HOME directory which we need later.

### Linux (Ubuntu 20.04) android-sdk install

```
# install python and Android sdk
sudo apt-get install android-sdk python3

# set Android home
export ANDROID_HOME=/usr/lib/android-sdk

# give user ownership of ANDROID_HOME path
sudo chown -R ubuntu:ubuntu $ANDROID_HOME
```

### Mac android-sdk install

```
# install python and Android sdk
brew install android-sdk python3

# set Android home
export ANDROID_HOME=/Users/$USER/Library/Android/sdk

# give user ownership of ANDROID_HOME path
sudo chown -R $USER $ANDROIANDROID_HOMED_SDK_ROOT 
```

## Install Command line only tools

Find latest command line only tools: [https://developer.android.com/studio#downloads](https://developer.android.com/studio#downloads:~:text=Command%20line%20tools%20only)

```
# create directory for Android command line tools
mkdir -p $ANDROID_HOME/cmdline-tools
cd $ANDROID_HOME/cmdline-tools

# From link above, set ANDROID_CLT equal to the name of Linux's latest SDK tools package release
export ANDROID_CLT=commandlinetools-linux-8512546_latest.zip

# Download Android command line tools and remove the zip file once finished unzipping
curl -O https://dl.google.com/android/repository/${ANDROID_CLT}
unzip $ANDROID_CLT && rm $ANDROID_CLT

# renames the command line tools directory that just was unzipped
mv cmdline-tools tools
```

## Install Android NDK

See GitHub actions to ensure latest NDK_VERSION: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml)

```
# set Android NDK version and install it
export NDK_VERSION=21.1.6352462
echo "y" | sudo ${ANDROID_HOME}/cmdline-tools/tools/bin/sdkmanager --install "ndk;${NDK_VERSION}"
```

## Build PyTorch Android native binary from source

See: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_s3_pytorch_android.yml)

```
# cd into whatever directory holds your djl directory
export PYTORCH_VERSION=1.11.0
export ANDROID_NDK=${ANDROID_HOME}/ndk/${NDK_VERSION}

# clone PyTorch to local machine
git clone https://github.com/pytorch/pytorch -b "v${PYTORCH_VERSION}" --recursive
cd pytorch

# need to build all four ABIs
export BUILD_LITE_INTERPRETER=0
./scripts/build_pytorch_android.sh arm64-v8a,armeabi-v7a,x86,x86_64

# zip all four ABIs
cd build_android_arm64-v8a && zip -r ~/arm64-v8a_native.zip install/include lib && cd ..
cd build_android_armeabi-v7a && zip -r ~/armeabi-v7a_native.zip install/include lib && cd ..
cd build_android_x86 && zip -r ~/x86_native.zip install/include lib && cd ..
cd build_android_x86_64 && zip -r ~/x86_64_native.zip install/include lib && cd ..
```

The zip files will be uploaded to S3 bucket in CI build

## Build PyTorch Android JNI

See: [https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_jni_s3_pytorch_android.yml](https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/native_jni_s3_pytorch_android.yml)

This command unzips all the files we zipped in the previous code block. It puts them into the directories where the DJL build expects to find them when it compiles.

```
cd ../djl/engines/pytorch/pytorch-native

# to avoid download PyTorch native from S3, manually unzip PyTorch native
mkdir libtorch_android && cd libtorch_android

# make directory for each Android ABI and unzip files
mkdir arm64-v8a armeabi-v7a x86 x86_64
cd arm64-v8a && unzip ~/arm64-v8a_native.zip && mv install/include . && cd ..
cd armeabi-v7a && unzip ~/armeabi-v7a_native.zip && mv install/include . && cd ..
cd x86 && unzip ~/x86_native.zip && mv install/include . && cd ..
cd x86_64 && unzip ~/x86_64_native.zip && mv install/include . && cd ..

# switch back to PyTorch native directory to build with gradle
cd ..
./gradlew compileAndroidJNI -Ppt_version=${PYTORCH_VERSION}
```

`jnilib/0.18.0/android` folder will be created after build, and shared library will be uploaded to S3 in CI build

## Build PyTorch android library (.aar) and publish to Sonatype snapshot repo

See: [ https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/publish_android_packages.yml]( https://github.com/deepjavalibrary/djl/blob/master/.github/workflows/publish_android_packages.yml)

The final command in this code block `./gradlew pTML` is optional. It stores a local copy of the DJL snapshot in your maven directory. If not done, then the app will pull the snapshot release of DJL from Sonatype. 

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

`ai.djl.android:pytorch-native:0.18.0-SNAPSHOT`  will be published to mavenLocal (your local ~/.m2 folder) if you ran `./gradlew pTML`

## Build PyTorch Android demo projects

See: [https://github.com/deepjavalibrary/djl-demo/tree/master/android/pytorch_android](https://github.com/deepjavalibrary/djl-demo/tree/master/android/pytorch_android)

From Android Studio, with an emulator turned on, run the following commands

```
cd djl-demo/android/pytorch_android/style_transfer_cyclegan
./gradlew iD
```
