plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(project(":api"))
    api(project(":basicdataset"))
    api(project(":model-zoo"))
    testImplementation(libs.slf4j.simple)

    testImplementation(project(":testing"))
    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }

    // Current engines and model zoos used for inference
    // runtimeOnly project(":engines:pytorch:pytorch-engine")
    // runtimeOnly project(":engines:pytorch:pytorch-model-zoo")
    // runtimeOnly project(":engines:tensorflow:tensorflow-engine")
    // runtimeOnly project(":engines:onnxruntime:onnxruntime-engine")
    // runtimeOnly "com.microsoft.onnxruntime:onnxruntime:${onnxruntime_version}"
    runtimeOnly(project(":engines:mxnet:mxnet-engine"))
    runtimeOnly(project(":engines:mxnet:mxnet-model-zoo"))
}
