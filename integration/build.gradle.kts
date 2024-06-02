plugins {
    ai.djl.javaProject
    application
    jacoco
}

dependencies {
    implementation(libs.commons.cli)
    implementation(libs.apache.log4j.slf4j)
    implementation(project(":basicdataset"))
    implementation(project(":model-zoo"))
    implementation(project(":testing"))

    // Don't use MXNet for aarch64
    if (arch != "aarch64")
        runtimeOnly(project(":engines:mxnet:mxnet-model-zoo"))

    runtimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    runtimeOnly(project(":engines:pytorch:pytorch-jni"))
    runtimeOnly(project(":engines:tensorflow:tensorflow-model-zoo"))
    runtimeOnly(project(":engines:ml:xgboost"))
    runtimeOnly(project(":engines:ml:lightgbm"))
    runtimeOnly(project(":engines:onnxruntime:onnxruntime-engine"))
    runtimeOnly(project(":extensions:tokenizers"))
}

tasks {
    compileJava {
        options.apply {
            release = 11
        }
    }

    register<Copy>("copyDependencies") {
        into("build/dependencies")
        from(configurations.runtimeClasspath)
    }

    run.configure {
        for (prop in systemProperties.iterator()) {
            if (prop.key.startsWith("ai.djl.") || prop.key == "nightly") {
                systemProperty(prop.key, prop.value)
            }
        }
        environment("TF_CPP_MIN_LOG_LEVEL" to "1") // turn off TensorFlow print out
        jvmArgs("-Xverify:none")
    }

    register<JavaExec>("debugEnv") {
        for (prop in systemProperties.iterator()) {
            if (prop.key.startsWith("ai.djl.") || prop.key == "nightly") {
                systemProperty(prop.key, prop.value)
            }
        }
        classpath = sourceSets.main.get().runtimeClasspath
        mainClass = "ai.djl.integration.util.DebugEnvironment"
    }

    distTar { enabled = false }
}

application {
    mainClass = System.getProperty("main", "ai.djl.integration.IntegrationTest")
}