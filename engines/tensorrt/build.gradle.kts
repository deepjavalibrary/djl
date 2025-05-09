plugins {
    ai.djl.javaProject
    ai.djl.cppFormatter
    ai.djl.publish
}

group = "ai.djl.tensorrt"

dependencies {
    api(project(":api"))

    testImplementation(libs.testng)
    testImplementation(libs.slf4j.simple)
    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-engine"))
}

open class Cmd @Inject constructor(@Internal val execOperations: ExecOperations) : DefaultTask()

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        inputs.properties(
            mapOf(
                "djlVersion" to libs.versions.djl, "trtVersion" to libs.versions.tensorrt.get(),
                "version" to version
            )
        )
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        outputs.dir(file("${baseResourcePath}/native/lib"))
        val trtVersion = libs.versions.tensorrt.get()
        val djlVersion = libs.versions.djl.get()
        val url = "https://publish.djl.ai/tensorrt/${trtVersion}/jnilib/${djlVersion}"
        val files = listOf("linux-x86_64/libdjl_trt.so")
        val jnilibDir = project.projectDir / "jnilib/${djlVersion}"

        val logger = project.logger
        doLast {
            for (entry in files) {
                val file = jnilibDir / entry
                if (file.exists()) {
                    logger.lifecycle("prebuilt or cached file found for $entry")
                } else if (!project.hasProperty("jni")) {
                    logger.lifecycle("Downloading $url/$entry")
                    file.parentFile.mkdirs()
                    "$url/$entry".url into file
                }
            }

            copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }
        }

        filesMatching("**/tensorrt.properties") {
            expand(mapOf("trtVersion" to libs.versions.tensorrt.get(), "version" to version))
        }
    }

    register<Cmd>("compileJNI") {
        val dir = project.projectDir
        doFirst {
            if ("linux" in os) {
                execOperations.exec {
                    workingDir = dir
                    commandLine("bash", "build.sh")
                }
            } else {
                throw IllegalStateException("Unknown Architecture $osName")
            }

            // for nightly ci
            val classifier = "${os}-x86_64"
            val ciDir = dir / "jnilib/${libs.versions.djl.get()}/${classifier}"
            copy {
                from(fileTree(buildDirectory) {
                    include("libdjl_trt.*")
                })
                into(ciDir)
            }
            delete("$home/.djl.ai/tensorrt")
        }
    }

    clean {
        doFirst {
            delete("$home/.djl.ai/tensorrt")
        }
    }
}
