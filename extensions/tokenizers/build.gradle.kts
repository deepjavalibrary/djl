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

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val djlVersion = libs.versions.djl.get()
        val tokenizersVersion = libs.versions.tokenizers.get()
        inputs.properties(
            mapOf(
                "djlVersion" to djlVersion,
                "tokenizersVersion" to tokenizersVersion,
                "version" to version
            )
        )
        val baseResourcePath = "${projectDir}/build/resources/main"
        outputs.dirs(
            File("${baseResourcePath}/native/lib"),
            File("${baseResourcePath}/nlp"),
            File("${baseResourcePath}/cv")
        )

        val logger = project.logger
        val dir = projectDir
        val hasJni = project.hasProperty("jni")
        val injected = project.objects.newInstance<InjectedOps>()
        val version = project.version

        doLast {
            var url = "https://publish.djl.ai/tokenizers"
            val files = mapOf(
                "win-x86_64/cpu/libwinpthread-1.dll" to "extra/win-x86_64/libwinpthread-1.dll",
                "win-x86_64/cpu/libgcc_s_seh-1.dll" to "extra/win-x86_64/libgcc_s_seh-1.dll",
                "win-x86_64/cpu/libstdc%2B%2B-6.dll" to "extra/win-x86_64/libstdc%2B%2B-6.dll",
                "win-x86_64/cpu/tokenizers.dll" to "$tokenizersVersion/jnilib/$djlVersion",
                "linux-x86_64/cpu/libtokenizers.so" to "$tokenizersVersion/jnilib/$djlVersion",
                "linux-aarch64/cpu/libtokenizers.so" to "$tokenizersVersion/jnilib/$djlVersion",
                "osx-aarch64/cpu/libtokenizers.dylib" to "$tokenizersVersion/jnilib/$djlVersion"
            )
            val jnilibDir = dir / "jnilib/$djlVersion"
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
            injected.fs.copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }

            url = "https://mlrepo.djl.ai/model"
            val tasks = listOf(
                "cv/zero_shot_image_classification",
                "cv/zero_shot_object_detection",
                "nlp/fill_mask",
                "nlp/question_answer",
                "nlp/text_classification",
                "nlp/text_embedding",
                "nlp/token_classification",
                "nlp/zero_shot_classification"
            )
            val prefix = File(baseResourcePath)
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

                if (task !in arrayOf("nlp/text_embedding", "nlp/text_classification"))
                    continue

                file = prefix / task / "ai.djl.huggingface.rust.json"
                if (file.exists())
                    logger.lifecycle("Rust model zoo metadata already exists: $task")
                else {
                    logger.lifecycle("Downloading Rust model zoo metadata: $task")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$task/ai/djl/huggingface/rust/models.json.gz".url
                    downloadPath gzipInto file
                }
            }
        }

        filesMatching("**/tokenizers.properties") {
            expand(mapOf("tokenizersVersion" to tokenizersVersion, "version" to version))
        }
    }

    register("compileJNI") {
        val djlVersion = libs.versions.djl.get()
        val dir = project.projectDir
        val buildDir = buildDirectory
        val injected = project.objects.newInstance<InjectedOps>()
        val flavor = flavor
        doFirst {
            if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                injected.exec.exec {
                    workingDir = dir
                    commandLine("bash", "build.sh", arch, flavor)
                }
            } else
                injected.exec.exec {
                    workingDir = dir
                    commandLine("${dir}/build.cmd")
                }

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${djlVersion}/"
            injected.fs.copy {
                from(buildDir / "jnilib")
                into(ciDir)
            }
            injected.fs.delete { delete("$home/.djl.ai/tokenizers") }
        }
    }

    register("compileAndroidJNI") {
        val djlVersion = libs.versions.djl.get()
        val dir = project.projectDir
        val buildDir = buildDirectory
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            for (abi in listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")) {
                injected.exec.exec {
                    workingDir = dir
                    commandLine("bash", "build_android.sh", abi)
                }
                val ciDir = dir / "jnilib/${djlVersion}/android/$abi"
                injected.fs.copy {
                    from(buildDir / "jnilib" / abi)
                    into(ciDir)
                }
                injected.fs.delete { delete("$buildDir/jnilib") }
            }
        }
    }

    register<Exec>("formatPython") {
        workingDir = project.projectDir
        commandLine("bash", "-c", "find . -name '*.py' -print0 | xargs -0 yapf --in-place")
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        val dir = projectDir
        doFirst {
            injected.fs.delete { delete("$home/.djl.ai/tokenizers") }
            injected.fs.delete { delete("jnilib") }
            injected.fs.delete { delete("rust/target") }
            injected.fs.delete { delete("src/main/python/build/") }
            injected.fs.delete { delete("src/main/python/dist/") }
            injected.fs.delete { delete("src/main/python/__pycache__/") }
            injected.fs.delete { delete("src/main/python/djl_converter.egg-info/") }
            injected.fs.delete { delete("src/main/python/djl_converter/__pycache__/") }

            val initFile = dir / "src/main/python/djl_converter/__init__.py"
            initFile.text = initFile.text.replace(Regex("\\n*__version__.*"), "\n")
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

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations

    @get:Inject
    val exec: ExecOperations
}
