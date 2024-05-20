import java.net.URLDecoder

plugins {
    ai.djl.javaProject
    ai.djl.publish
    ai.djl.cppFormatter
}

group = "ai.djl.huggingface"

dependencies {
    api(project(":api"))

    testImplementation(project(":engines:pytorch:pytorch-engine"))
    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.dir(buildDirectory / "classes/java/main/native/lib")
        doLast {
            var url = "https://publish.djl.ai/tokenizers"
            val (tokenizers, djl) = libs.versions.tokenizers.get() to libs.versions.djl.get()
            val files = mapOf(
                "win-x86_64/libwinpthread-1.dll" to "extra",
                "win-x86_64/libgcc_s_seh-1.dll" to "extra",
                "win-x86_64/libstdc%2B%2B-6.dll" to "extra",
                "win-x86_64/tokenizers.dll" to "$tokenizers/jnilib/$djl",
                "linux-x86_64/libtokenizers.so" to "$tokenizers/jnilib/$djl",
                "linux-aarch64/libtokenizers.so" to "$tokenizers/jnilib/$djl",
                "osx-x86_64/libtokenizers.dylib" to "$tokenizers/jnilib/$djl",
                "osx-aarch64/libtokenizers.dylib" to "$tokenizers/jnilib/$djl"
            )
            val jnilibDir = project.projectDir / "jnilib/$djl"
            for ((key, value) in files) {
                val file = jnilibDir / URLDecoder.decode(key, "UTF-8")
                if (file.exists())
                    project.logger.lifecycle("prebuilt or cached file found for $key")
                else if ("extra" == value || !project.hasProperty("jni")) {
                    project.logger.lifecycle("Downloading $url/$value/$key")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$value/$key".url
                    downloadPath into file
                }
            }
            copy {
                from(jnilibDir)
                into(buildDirectory / "classes/java/main/native/lib")
            }

            // write properties
            val propFile = buildDirectory / "classes/java/main/native/lib/tokenizers.properties"
            propFile.text = "version=$tokenizers-$version\n"

            url = "https://mlrepo.djl.ai/model/nlp"
            val tasks = listOf(
                "fill_mask",
                "question_answer",
                "text_classification",
                "text_embedding",
                "token_classification"
            )
            val prefix = buildDirectory / "classes/java/main/nlp"
            for (task in tasks) {
                var file = prefix / task / "ai.djl.huggingface.pytorch.json"
                if (file.exists())
                    project.logger.lifecycle("PyTorch model zoo metadata alrady exists: $task")
                else {
                    project.logger.lifecycle("Downloading PyTorch model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$task/ai/djl/huggingface/pytorch/models.json.gz".url
                    downloadPath gzipInto file
                }

                if ("text_embedding" != task)
                    continue

                file = prefix / task / "ai.djl.huggingface.rust.json"
                if (file.exists())
                    project.logger.lifecycle("Rust model zoo metadata alrady exists: $task")
                else {
                    project.logger.lifecycle("Downloading Rust model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$task/ai/djl/huggingface/rust/models.json.gz".url
                    downloadPath gzipInto file
                }
            }
        }
    }

    register("compileJNI") {
        doFirst {
            if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                exec {
                    commandLine("bash", "build.sh", libs.versions.tokenizers.get(), arch)
                }
            } else
                exec {
                    commandLine("${project.projectDir}/build.cmd", libs.versions.tokenizers.get())
                }

            // for ci to upload to S3
            val ciDir = project.projectDir / "jnilib/${libs.versions.djl.get()}/"
            copy {
                from(buildDirectory / "jnilib")
                into(ciDir)
            }
            delete("$home/.djl.ai/tokenizers")
        }
    }

    register("formatPython") {
        doFirst {
            exec {
                commandLine("bash", "-c", "find . -name '*.py' -print0 | xargs -0 yapf --in-place")
            }
        }
    }

    clean {
        doFirst {
            delete("$home/.djl.ai/tokenizers")
            delete("rust/target")
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL NLP utilities for Huggingface tokenizers"
                description = "Deep Java Library (DJL) NLP utilities for Huggingface tokenizers"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}