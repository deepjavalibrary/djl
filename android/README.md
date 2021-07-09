# DJL Android

DJL Android allows you to run inference with Android devices.
It has the following two modules:

- Core package: contains some Image processing toolkit for Android user using DJL
- PyTorch Native: contains DJL PyTorch Android native package

## Installation
You need to have Android SDK and Android NDK installed on your machine.

The minimum API level for DJL Android is 26.

In gradle, you can include the snapshot repository and add the 4 modules in your dependencies:

```
dependencies {
    implementation "ai.djl:api:0.12.0"
    implementation "ai.djl.android:core:0.12.0"
    androidRuntimeOnly "ai.djl.pytorch:pytorch-engine:0.12.0"
    androidRuntimeOnly "ai.djl.android:pytorch-native:0.12.0"
}
```
