# DJL Android

DJL Android allows you to run inference with Android devices.
It has the following two modules:

- Core package: contains some Image processing toolkit for Android user using DJL
- PyTorch Native: contains DJL PyTorch Android native package
- ONNXRuntime: contains DJL ONNXRuntime Android package

## Installation
You need to have Android SDK and Android NDK installed on your machine.

The minimum API level for DJL Android is 26.

In gradle, you can add the 5 modules in your dependencies:

```
dependencies {
    implementation platform("ai.djl:bom:0.21.0")

    implementation "ai.djl:api"
    // TODO: remove djl android package version in 0.21.0
    implementation "ai.djl.android:core:0.21.0"
    runtimeOnly "ai.djl.pytorch:pytorch-engine"
    runtimeOnly "ai.djl.android:pytorch-native:0.21.0"
    runtimeOnly "ai.djl.android:onnxruntime:0.21.0"
}
```
