plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(projects.api)
    api(projects.basicdataset)
    api(projects.modelZoo)
    testImplementation(libs.slf4j.simple)

    testImplementation(projects.testing)
    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }

    // Current engines and model zoos used for inference
    // runtimeOnly project(":engines:pytorch:pytorch-engine")
    // runtimeOnly project(":engines:pytorch:pytorch-model-zoo")
    // runtimeOnly project(":engines:tensorflow:tensorflow-engine")
    // runtimeOnly project(":engines:onnxruntime:onnxruntime-engine")
    // runtimeOnly "com.microsoft.onnxruntime:onnxruntime:${onnxruntime_version}"
    runtimeOnly(projects.engines.mxnet.mxnetEngine)
    runtimeOnly(projects.engines.mxnet.mxnetModelZoo)
}
