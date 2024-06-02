plugins {
    ai.djl.javaProject
    application
}

dependencies {
    implementation(libs.commons.cli)
    implementation(libs.apache.log4j.slf4j)
    implementation(project(":basicdataset"))
    implementation(project(":model-zoo"))
    implementation(project(":extensions:timeseries"))
    implementation(project(":extensions:tokenizers"))
    implementation(project(":extensions:audio"))

    runtimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    runtimeOnly(project(":engines:tensorflow:tensorflow-model-zoo"))
    runtimeOnly(project(":engines:mxnet:mxnet-model-zoo"))
    runtimeOnly(project(":engines:onnxruntime:onnxruntime-engine"))

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
}

application {
    mainClass = System.getProperty("main", "ai.djl.examples.inference.cv.ObjectDetection")
}

tasks {

    run.configure {
        environment("TF_CPP_MIN_LOG_LEVEL" to "1") // turn off TensorFlow print out
        for (prop in System.getProperties().iterator()) {
            val key = prop.key.toString()
            if (key.startsWith("ai.djl.")) {
                systemProperty(key, prop.value)
            }
        }
    }

    register<JavaExec>("listmodels") {
        for (prop in System.getProperties().iterator()) {
            val key = prop.key.toString()
            if (key.startsWith("ai.djl.")) {
                systemProperty(key, prop.value)
            }
        }
        if (!systemProperties.containsKey("ai.djl.logging.level")) {
            systemProperty("ai.djl.logging.level", "debug")
        }
        classpath = sourceSets.main.get().runtimeClasspath
        mainClass = "ai.djl.examples.inference.ListModels"
    }

    distTar { enabled = false }
    distZip { enabled = false }
}