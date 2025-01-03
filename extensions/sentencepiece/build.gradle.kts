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

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        inputs.properties(mapOf("djlVersion" to libs.versions.djl.get(), "sentencepieceVersion" to libs.versions.sentencepiece.get()))
        outputs.dir("$baseResourcePath/native/lib")
        val jnilibDir = project.projectDir / "jnilib/${libs.versions.djl.get()}"
        val logger = project.logger
        doLast {
            val url =
                "https://publish.djl.ai/sentencepiece-${libs.versions.sentencepiece.get()}/jnilib/${libs.versions.djl.get()}"
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
                else if (!project.hasProperty("jni")) {
                    logger.lifecycle("Downloading $url/$key")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$key/$value".url
                    downloadPath into file
                }
            }
            copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }
        }

        filesMatching("**/sentencepiece.properties") {
            expand(mapOf("sentencepieceVersion" to libs.versions.sentencepiece.get(), "version" to version))
        }
    }

    register<Cmd>("compileJNI") {
        val dir = project.projectDir
        doFirst {
            if ("win" in os)
                execOperations.exec {
                    workingDir = dir
                    commandLine("${dir}/build.cmd", "v${libs.versions.sentencepiece.get()}")
                }
            else if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                execOperations.exec {
                    workingDir = dir
                    commandLine("bash", "build.sh", "v${libs.versions.sentencepiece.get()}", arch)
                }
            } else
                throw IllegalStateException("Unknown Architecture $osName")

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${libs.versions.djl.get()}/"
            copy {
                from(buildDirectory / "jnilib")
                into(ciDir)
            }
            delete("$home/.djl.ai/sentencepiece")
        }
    }

    clean {
        doFirst {
            delete("$home/.djl.ai/sentencepiece")
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
