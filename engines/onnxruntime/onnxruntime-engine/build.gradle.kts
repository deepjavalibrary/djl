import java.net.URL

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.onnxruntime"

dependencies {
    api(projects.api)
    api(libs.onnxruntime)

    testImplementation(projects.testing)
    testImplementation(projects.engines.pytorch.pytorchEngine)
    testImplementation(projects.extension.tokenizers)

    testRuntimeOnly(libs.slf4j.simple)
}

tasks {
    processResources {
        outputs.dir(projectDir / "build/classes/java/main/nlp")
        doLast {
            val url = "https://mlrepo.djl.ai/model/nlp"
            val tasks = listOf("fill_mask",
                               "question_answer",
                               "text_classification",
                               "text_embedding",
                               "token_classification")
            val prefix = projectDir / "build/classes/java/main/nlp"
            for (task in tasks) {
                val file = prefix / task / "ai.djl.huggingface.onnxruntime.json"
                if (file.exists())
                    project.logger.lifecycle("model zoo metadata alrady exists: $task")
                else {
                    project.logger.lifecycle("Downloading model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    URL("$url/$task/ai/djl/huggingface/onnxruntime/models.json.gz") gzipInto file
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