rootProject.name = "djl"

include(":api")
include(":basicdataset")
include(":djl-zero")
include(":engines:llama")
include(":engines:ml:xgboost")
include(":engines:ml:lightgbm")
include(":engines:mxnet:jnarator")
include(":engines:mxnet:mxnet-engine")
include(":engines:mxnet:mxnet-model-zoo")
include(":engines:mxnet:native")
include(":engines:onnxruntime:onnxruntime-android")
include(":engines:onnxruntime:onnxruntime-engine")
include(":engines:pytorch:pytorch-engine")
include(":engines:pytorch:pytorch-jni")
include(":engines:pytorch:pytorch-model-zoo")
include(":engines:pytorch:pytorch-native")
include(":engines:tensorflow:tensorflow-api")
include(":engines:tensorflow:tensorflow-engine")
include(":engines:tensorflow:tensorflow-model-zoo")
include(":engines:tensorflow:tensorflow-native")
include(":engines:tensorrt")
include(":examples")
include(":extensions:audio")
include(":extensions:aws-ai")
include(":extensions:fasttext")
include(":extensions:hadoop")
include(":extensions:opencv")
include(":extensions:sentencepiece")
include(":extensions:tokenizers")
include(":extensions:tablesaw")
include(":extensions:timeseries")
if (JavaVersion.current() < JavaVersion.VERSION_21)
    include(":extensions:spark")
include(":integration")
include(":model-zoo")
include(":testing")
include(":jacoco")

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven("https://oss.sonatype.org/content/repositories/snapshots/")
    }
}

// we need to disable this because of this bug: https://github.com/gradle/gradle/issues/19254
// Method org/gradle/accessors/dm/RootProjectAccessor_Decorated.getExtensions()Lorg/gradle/api/plugins/ExtensionContainer; is abstract
//enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")