plugins {
    ai.djl.javaProject
    ai.djl.publish
    ai.djl.cppFormatter
}

group = "ai.djl.fasttext"

dependencies {
    api(project(":api"))

    testImplementation(project(":basicdataset"))
    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        inputs.properties(mapOf("djlVersion" to libs.versions.djl.get(),
            "fasttextVersion" to libs.versions.fasttext.get(),
            "version" to version))
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        outputs.dir("$baseResourcePath/native/lib")
        val jnilibDir = project.projectDir / "jnilib/${libs.versions.djl.get()}"
        val logger = project.logger
        val hasJni = project.hasProperty("jni")
        doLast {
            val url =
                "https://publish.djl.ai/fasttext-${libs.versions.fasttext.get()}/jnilib/${libs.versions.djl.get()}"
            val files = mapOf(
                "linux-x86_64" to "libjni_fasttext.so",
                "linux-aarch64" to "libjni_fasttext.so",
                "osx-aarch64" to "libjni_fasttext.dylib"
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
            copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }
        }

        filesMatching("**/fasttext.properties") {
            expand(mapOf("fasttextVersion" to libs.versions.fasttext.get(), "version" to version))
        }
    }

    register<Cmd>("compileJNI") {
        val dir = project.projectDir
        doFirst {
            check("mac" in os || "linux" in os) { "Unknown Architecture $osName" }
            execOperations.exec {
                workingDir = dir
                commandLine("bash", "build.sh")
            }

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${libs.versions.djl.get()}"
            copy {
                from(buildDirectory / "jnilib")
                into(ciDir)
            }
            delete("$home/.djl.ai/fasttext")
        }
    }

    clean {
        doFirst {
            delete("$home/.djl.ai/fasttext")
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                artifactId = "fasttext-engine"
                name = "Fasttext Engine Adapter"
                description = "Fasttext Engine Adapter for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
