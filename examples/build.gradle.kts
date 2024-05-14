@file:Suppress("UNCHECKED_CAST")

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
        // @Niels Doucet
        // Just a heads-up: gradle support warned me about systemProperties System.getProperties(). It's really
        // dangerous to just copy over all system properties to a task invocation. You should really be specific about
        // the properties you'd like to expose inside the task, or you might get very strange issues.
        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
        systemProperty("file.encoding", "UTF-8")
    }

    register<JavaExec>("listmodels") {
        systemProperties(System.getProperties() as Map<String, Any>)
        systemProperties.remove("user.dir")
        systemProperty("file.encoding", "UTF-8")
        classpath = sourceSets.main.get().runtimeClasspath
        mainClass = "ai.djl.examples.inference.ListModels"
    }
    distTar { enabled = false }
    distZip { enabled = false }
}