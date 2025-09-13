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

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val djlVersion = libs.versions.djl.get()
        val fasttextVersion = libs.versions.fasttext.get()
        inputs.properties(
            mapOf(
                "djlVersion" to djlVersion,
                "fasttextVersion" to fasttextVersion,
                "version" to version
            )
        )
        val baseResourcePath = "${project.projectDir}/build/resources/main"
        outputs.dir("$baseResourcePath/native/lib")
        val jnilibDir = project.projectDir / "jnilib/${djlVersion}"
        val logger = project.logger
        val hasJni = project.hasProperty("jni")
        val version = project.version
        val injected = project.objects.newInstance<InjectedOps>()

        doLast {
            val url = "https://publish.djl.ai/fasttext-${fasttextVersion}/jnilib/${djlVersion}"
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
            injected.fs.copy {
                from(jnilibDir)
                into("$baseResourcePath/native/lib")
            }
        }

        filesMatching("**/fasttext.properties") {
            expand(mapOf("fasttextVersion" to fasttextVersion, "version" to version))
        }
    }

    register("compileJNI") {
        val djlVersion = libs.versions.djl.get()
        val dir = projectDir
        val buildDir = buildDirectory
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            check("mac" in os || "linux" in os) { "Unknown Architecture $osName" }
            injected.exec.exec {
                workingDir = dir
                commandLine("bash", "build.sh")
            }

            // for ci to upload to S3
            val ciDir = dir / "jnilib/${djlVersion}"
            injected.fs.copy {
                from(buildDir / "jnilib")
                into(ciDir)
            }
            injected.fs.delete { delete("$home/.djl.ai/fasttext") }
        }
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        doFirst {
            injected.fs.delete { delete("$home/.djl.ai/fasttext") }
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

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations

    @get:Inject
    val exec: ExecOperations
}
