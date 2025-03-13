plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.onnxruntime"

dependencies {
    api(project(":api"))
    api(libs.onnxruntime)

    testImplementation(project(":testing"))
    testImplementation(project(":engines:pytorch:pytorch-engine"))
    testImplementation(project(":extensions:tokenizers"))
    testImplementation("com.microsoft.onnxruntime:onnxruntime-extensions:${libs.versions.onnxruntimeExtensions.get()}")

    testRuntimeOnly(libs.slf4j.simple)
}

tasks {
    val basePath = "${project.projectDir}/build/resources/main/nlp"
    val logger = project.logger
    processResources {
        outputs.dir(basePath)
        doLast {
            val url = "https://mlrepo.djl.ai/model/nlp"
            val tasks = listOf(
                "fill_mask",
                "question_answer",
                "text_classification",
                "text_embedding",
                "token_classification",
                "zero_shot_classification"
            )
            for (task in tasks) {
                val file = File("$basePath/$task/ai.djl.huggingface.onnxruntime.json")
                if (file.exists())
                    logger.lifecycle("model zoo metadata already exists: $task")
                else {
                    logger.lifecycle("Downloading model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    "$url/$task/ai/djl/huggingface/onnxruntime/models.json.gz".url gzipInto file
                }
            }
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            artifactId = "onnxruntime-engine"
            pom {
                name = "DJL Engine Adapter for ONNX Runtime"
                description = "Deep Java Library (DJL) Engine Adapter for ONNX Runtime"
                url = "http://www.djl.ai/engines/onnxruntime/${project.name}"
            }
        }
    }
}
