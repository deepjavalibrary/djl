plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.android"

dependencies {
    api(project(":engines:onnxruntime:onnxruntime-engine")) {
        exclude("com.microsoft.onnxruntime", "onnxruntime")
    }
    api(libs.onnxruntimeAndroid)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            artifactId = "onnxruntime"
            pom {
                name = "DJL ONNX Runtime engine for Android"
                description = "Deep Java Library (DJL) Engine Adapter for ONNX Runtime"
                url = "http://www.djl.ai/engines/onnxruntime/${project.name}"
            }
        }
    }
}
