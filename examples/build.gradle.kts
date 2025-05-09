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
    runtimeOnly("com.microsoft.onnxruntime:onnxruntime-extensions:${libs.versions.onnxruntimeExtensions.get()}")

    testImplementation(libs.testng)
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

    distTar { enabled = false }
    distZip { enabled = false }

    sourceSets.main.get().java.files
        .filter { it.text.contains("public static void main(String[] args)") }
        .map {
            it.path.substringAfter("java${File.separatorChar}").replace(File.separatorChar, '.')
                .substringBefore(".java")
        }
        .forEach { className ->
            val taskName = className.substringAfterLast(".")

            register<JavaExec>(name = taskName) {
                classpath = sourceSets.main.get().runtimeClasspath
                mainClass = className
                group = "application"
                for (prop in System.getProperties().iterator()) {
                    val key = prop.key.toString()
                    if (key.startsWith("ai.djl.")) {
                        systemProperty(key, prop.value)
                    }
                }
                if (!systemProperties.containsKey("ai.djl.logging.level")) {
                    systemProperty("ai.djl.logging.level", "debug")
                }

                allJvmArgs = allJvmArgs + listOf(
                    // kryo compatability
                    // from https://github.com/EsotericSoftware/kryo/blob/cb255af4f8df4f539778a325b8b4836d41f84de9/pom.xml#L435
                    "--add-opens=java.base/java.lang=ALL-UNNAMED",
                    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
                    "--add-opens=java.base/java.net=ALL-UNNAMED",
                    "--add-opens=java.base/java.nio=ALL-UNNAMED",
                    "--add-opens=java.base/java.time=ALL-UNNAMED",
                    "--add-opens=java.base/java.util=ALL-UNNAMED",
                    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
                    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
                    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
                )
            }
        }
}

