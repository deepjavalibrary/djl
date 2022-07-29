
# ONNXRuntime Android Build Procedure

This procedure is meant only for developers who want to build PyTorch native binaries from source, this is not intended for regular users.

Follow this setup guide in order to run DJL apps on an Android. In order to successfully run the apps, you must install several Android items and build Android ABIs outlined below.

## Prerequisites

```
# Run the following command (assume you have python3 installed already)
export PYTHON=python3

# install needed python libraries
python3 -m pip install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions
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
brew install android-sdk

# set Android home
export ANDROID_HOME=/Users/$USER/Library/Android/sdk

# give user ownership of ANDROID_HOME path
sudo chown -R $USER $ANDROID_HOME 
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

## Build ONNXRuntime android library (.aar) and publish to local output folder

```
# move into djl/android directory
cd android 

# for linux user
./gradlew :onnxruntime-native:assembleDebug

# for windows user
.\gradlew.bat :onnxruntime-native:assembleDebug

```
`onnxruntime-native-debug.aar`  will be published to your local ~/djl/android/onnxruntime-native/build/outputs/aar folder
