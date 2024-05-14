import java.net.URL

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
        outputs.dir(layout.buildDirectory / "classes/java/main/native/lib")
        doLast {
            val url = "https://publish.djl.ai/fasttext-${libs.versions.fasttext.get()}/jnilib/${libs.versions.djl.get()}"
            val files = mapOf("linux-x86_64" to "libjni_fasttext.so",
                              "osx-x86_64" to "libjni_fasttext.dylib",
                              "osx-aarch64" to "libjni_fasttext.dylib")
            val jnilibDir = project.projectDir / "jnilib/${libs.versions.djl.get()}"
            for ((key, value) in files) {
                val file = jnilibDir / key / value
                if (file.exists())
                    project.logger.lifecycle("prebuilt or cached file found for $value")
                else if (!project.hasProperty("jni")) {
                    project.logger.lifecycle("Downloading $url/$key")
                    file.parentFile.mkdirs()
                    val downloadPath = "$url/$key/$value".url
                    downloadPath into file
                }
            }
            copy {
                from(jnilibDir)
                into(layout.buildDirectory / "classes/java/main/native/lib")
            }

            // write properties
            val propFile = layout.buildDirectory / "classes/java/main/native/lib/fasttext.properties"
            propFile.text = "version=${libs.versions.fasttext.get()}-${version}\n"
        }
    }

    register("compileJNI") {
        doFirst {
            check("mac" in os || "linux" in os) { "Unknown Architecture $osName" }
            exec {
                commandLine("bash", "build.sh")
            }

            // for ci to upload to S3
            val ciDir = project.projectDir / "jnilib/${libs.versions.djl.get()}"
            copy {
                from(layout.buildDirectory / "jnilib")
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