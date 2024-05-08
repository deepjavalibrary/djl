@file:Suppress("UNCHECKED_CAST")

plugins {
    ai.djl.javaProject
    application
}

dependencies {
    implementation(libs.commons.cli)
    implementation(libs.apache.log4j.slf4j)
    implementation(projects.basicdataset)
    implementation(projects.modelZoo)
    implementation(projects.extension.timeseries)
    implementation(projects.extension.tokenizers)
    implementation(projects.extension.audio)

    runtimeOnly(projects.engines.pytorch.pytorchModelZoo)
    runtimeOnly(projects.engines.tensorflow.tensorflowModelZoo)
    runtimeOnly(projects.engines.mxnet.mxnetModelZoo)
    runtimeOnly(projects.engines.onnxruntime.onnxruntimeEngine)

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