@file:Suppress("UNCHECKED_CAST")

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
        // need to use `project` as receiver otherwise something else will be picked up
        javaCompiler = project.javaToolchains.compilerFor { languageVersion = JavaLanguageVersion.of(11) }
        // you cant remove from `options.compilerArgs`, just assign a new value to it
        options.apply { compilerArgs = compilerArgs - listOf("--release", "8") }
    }

    register<Copy>("copyDependencies") {
        into("build/dependencies")
        from(configurations.runtimeClasspath)
    }

    run.configure {
        environment("TF_CPP_MIN_LOG_LEVEL" to "1") // turn off TensorFlow print out
        // @Niels Doucet
        // Just a heads-up: gradle support warned me about systemProperties System.getProperties(). It's really
        // dangerous to just copy over all system properties to a task invocation. You should really be specific about
        // the properties you'd like to expose inside the task, or you might get very strange issues.
        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
        systemProperty("file.encoding", "UTF-8")
        jvmArgs("-Xverify:none")
    }

    register<JavaExec>("debugEnv") {
        classpath = sourceSets.main.get().runtimeClasspath
        systemProperties = System.getProperties().toMap() as Map<String, Any>
        systemProperties.remove("user.dir")
        systemProperties["ai.djl.logging.level"] = "debug"
        mainClass = "ai.djl.integration.util.DebugEnvironment"
    }

    distTar { enabled = false }
}

application {
    mainClass = System.getProperty("main", "ai.djl.integration.IntegrationTest")
}
