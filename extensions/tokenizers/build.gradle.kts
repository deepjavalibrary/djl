import java.net.URLDecoder

plugins {
    ai.djl.javaProject
    ai.djl.publish
    ai.djl.cppFormatter
}

group = "ai.djl.huggingface"

val flavor = when {
    project.hasProperty("cuda") -> project.property("cuda").toString()
    else -> "cpu"
}

dependencies {
    api(project(":api"))

    testImplementation(project(":engines:pytorch:pytorch-engine"))
    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        inputs.properties(
            mapOf(
                "djlVersion" to libs.versions.djl.get(),
                "tokenizersVersion" to libs.versions.tokenizers.get(),
                "version" to version
            )
        )
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        outputs.dirs(File("${baseResourcePath}/native/lib"), File("${baseResourcePath}/nlp"))

        val logger = project.logger
        val dir = project.projectDir
        val hasJni = project.hasProperty("jni")
        doLast {
            var url = "https://publish.djl.ai/tokenizers"
            val (tokenizers, djl) = libs.versions.tokenizers.get() to libs.versions.djl.get()
            val files = mapOf(
                "win-x86_64/cpu/libwinpthread-1.dll" to "extra/win-x86_64/libwinpthread-1.dll",
                "win-x86_64/cpu/libgcc_s_seh-1.dll" to "extra/win-x86_64/libgcc_s_seh-1.dll",
                "win-x86_64/cpu/libstdc%2B%2B-6.dll" to "extra/win-x86_64/libstdc%2B%2B-6.dll",
                "win-x86_64/cpu/tokenizers.dll" to "$tokenizers/jnilib/$djl",
                "linux-x86_64/cpu/libtokenizers.so" to "$tokenizers/jnilib/$djl",
                "linux-aarch64/cpu/libtokenizers.so" to "$tokenizers/jnilib/$djl",
                "osx-aarch64/cpu/libtokenizers.dylib" to "$tokenizers/jnilib/$djl"
            )
            val jnilibDir = dir / "jnilib/$djl"
            for ((key, value) in files) {
                val file = jnilibDir / URLDecoder.decode(key, "UTF-8")
                if (file.exists())
                    logger.lifecycle("prebuilt or cached file found for $key")
                else if (value.startsWith("extra")) {
                    logger.lifecycle("Downloading $url/$value")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$value".url
                    downloadPath into file
                } else if (!hasJni) {
                    logger.lifecycle("Downloading $url/$value/$key")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$value/$key".url
                    downloadPath into file
                }
            }
            copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }

            url = "https://mlrepo.djl.ai/model/nlp"
            val tasks = listOf(
                "fill_mask",
                "question_answer",
                "text_classification",
                "text_embedding",
                "token_classification"
            )
            val prefix = File("$baseResourcePath/nlp")
            for (task in tasks) {
                var file = prefix / task / "ai.djl.huggingface.pytorch.json"
                if (file.exists())
                    logger.lifecycle("PyTorch model zoo metadata already exists: $task")
                else {
                    logger.lifecycle("Downloading PyTorch model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$task/ai/djl/huggingface/pytorch/models.json.gz".url
                    downloadPath gzipInto file
                }

                if (task !in arrayOf("text_embedding", "text_classification"))
                    continue

                file = prefix / task / "ai.djl.huggingface.rust.json"
                if (file.exists())
                    logger.lifecycle("Rust model zoo metadata alrady exists: $task")
                else {
                    logger.lifecycle("Downloading Rust model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$task/ai/djl/huggingface/rust/models.json.gz".url
                    downloadPath gzipInto file
                }
            }
        }

        filesMatching("**/tokenizers.properties") {
            expand(mapOf("tokenizersVersion" to libs.versions.tokenizers.get(), "version" to version))
        }
    }

    register<Cmd>("compileJNI") {
        val dir = project.projectDir
        doFirst {
            if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                execOperations.exec {
                    workingDir = dir
                    commandLine("bash", "build.sh", arch, flavor)
                }
            } else
                execOperations.exec {
                    workingDir = dir
                    commandLine("${dir}/build.cmd")
                }

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${libs.versions.djl.get()}/"
            copy {
                from(buildDirectory / "jnilib")
                into(ciDir)
            }
            delete("$home/.djl.ai/tokenizers")
        }
    }

    register<Cmd>("compileAndroidJNI") {
        val dir = project.projectDir
        doFirst {
            for (abi in listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")) {
                execOperations.exec {
                    workingDir = dir
                    commandLine("bash", "build_android.sh", abi)
                }
                val ciDir = dir / "jnilib/${libs.versions.djl.get()}/android/$abi"
                copy {
                    from(buildDirectory / "jnilib" / abi)
                    into(ciDir)
                }
                delete("$buildDirectory/jnilib")
            }
        }
    }

    register<Exec>("formatPython") {
        workingDir = project.projectDir
        commandLine("bash", "-c", "find . -name '*.py' -print0 | xargs -0 yapf --in-place")
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
