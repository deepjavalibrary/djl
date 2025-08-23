plugins {
    ai.djl.javaProject
    ai.djl.publish
    ai.djl.cppFormatter
}

group = "ai.djl.sentencepiece"

dependencies {
    api(project(":api"))

    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val djlVersion = libs.versions.djl.get()
        val sentencepieceVersion = libs.versions.sentencepiece.get()
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        inputs.properties(
            mapOf(
                "djlVersion" to djlVersion,
                "sentencepieceVersion" to sentencepieceVersion
            )
        )
        outputs.dir("$baseResourcePath/native/lib")
        val jnilibDir = project.projectDir / "jnilib/${djlVersion}"
        val logger = project.logger
        val hasJni = project.hasProperty("jni")
        val injected = project.objects.newInstance<InjectedOps>()
        val version = project.version

        doLast {
            val url = "https://publish.djl.ai/sentencepiece-${sentencepieceVersion}/jnilib/${djlVersion}"
            val files = mapOf(
                "win-x86_64" to "sentencepiece_native.dll",
                "linux-x86_64" to "libsentencepiece_native.so",
                "linux-aarch64" to "libsentencepiece_native.so",
                "osx-aarch64" to "libsentencepiece_native.dylib"
            )
            for ((key, value) in files) {
                val file = jnilibDir / key / value
                if (file.exists())
                    logger.lifecycle("prebuilt or cached file found for $value")
                else if (!hasJni) {
                    logger.lifecycle("Downloading $url/$key")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$key/$value".url
                    downloadPath into file
                }
            }
            injected.fs.copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }
        }

        filesMatching("**/sentencepiece.properties") {
            expand(mapOf("sentencepieceVersion" to sentencepieceVersion, "version" to version))
        }
    }

    register("compileJNI") {
        val dir = project.projectDir
        val buildDir = buildDirectory
        val djlVersion = libs.versions.djl.get()
        val sentencepieceVersion = libs.versions.sentencepiece.get()
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            if ("win" in os)
                injected.exec.exec {
                    workingDir = dir
                    commandLine("${dir}/build.cmd", "v${sentencepieceVersion}")
                }
            else if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                injected.exec.exec {
                    workingDir = dir
                    commandLine("bash", "build.sh", "v${sentencepieceVersion}", arch)
                }
            } else
                throw IllegalStateException("Unknown Architecture $osName")

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${djlVersion}/"
            injected.fs.copy {
                from(buildDir / "jnilib")
                into(ciDir)
            }
            injected.fs.delete { delete("$home/.djl.ai/sentencepiece") }
        }
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            injected.fs.delete { delete("$home/.djl.ai/sentencepiece") }
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL NLP utilities for SentencePiece"
                description = "Deep Java Library (DJL) NLP utilities for SentencePiece"
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
