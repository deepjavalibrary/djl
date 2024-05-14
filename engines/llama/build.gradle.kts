import java.net.URL

plugins {
    ai.djl.javaProject
    ai.djl.cppFormatter
    ai.djl.publish
}

group = "ai.djl.llama"

dependencies {
    api(project(":api"))

    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.dir(project.projectDir / "build/classes/java/main/native/lib")
        doLast {
            val llamacpp = libs.versions.llamacpp.get()
            val djl = libs.versions.djl.get()
            var url = "https://publish.djl.ai/llama/$llamacpp/jnilib/$djl"
            val files = listOf("linux-x86_64/libdjl_llama.so",
                               "linux-x86_64/libllama.so",
                               "linux-aarch64/libdjl_llama.so",
                               "linux-aarch64/libllama.so",
                               "osx-x86_64/libdjl_llama.dylib",
                               "osx-x86_64/libllama.dylib",
                               "osx-x86_64/ggml-metal.metal",
                               "osx-aarch64/libdjl_llama.dylib",
                               "osx-aarch64/libllama.dylib",
                               "osx-aarch64/ggml-metal.metal",
                               "win-x86_64/djl_llama.dll",
                               "win-x86_64/llama.dll")
            val jnilibDir = project.projectDir / "jnilib/$djl"
            files.forEach {
                val file = jnilibDir / it
                if (file.exists())
                    project.logger.lifecycle("prebuilt or cached file found for $it")
                else if (!project.hasProperty("jni")) {
                    project.logger.lifecycle("Downloading $url/$it")
                    file.parentFile.mkdirs()
                    "$url/$it".url into file
                }
            }
            copy {
                from(jnilibDir)
                into(project.projectDir / "build/classes/java/main/native/lib")
            }

            // write properties
            val propFile = project.projectDir / "build/classes/java/main/native/lib/llama.properties"
            propFile.text = "version=$llamacpp-$version\n"

            url = "https://mlrepo.djl.ai/model/nlp/text_generation/ai/djl/huggingface/gguf/models.json.gz"
            val prefix = project.projectDir / "build/classes/java/main/nlp/text_generation"
            val file = prefix / "ai.djl.huggingface.gguf.json"
            if (file.exists())
                project.logger.lifecycle("gguf index file already exists")
            else {
                project.logger.lifecycle("Downloading gguf index file")
                file.parentFile.mkdirs()
                url.url gzipInto file
            }
        }
    }

    publishing {
        publications {
            named<MavenPublication>("maven") {
                pom {
                    name = "DJL NLP utilities for Llama.cpp"
                    description = "Deep Java Library (DJL) NLP utilities for llama.cpp"
                    url = "http://www.djl.ai/engines/${project.name}"
                }
            }
        }
    }

    register("compileJNI") {
        doFirst {
            val cp = configurations.runtimeClasspath.get().resolve().joinToString(":")
            if ("mac" in os || "linux" in os) {
                val arch = if (arch == "amd64") "x86_64" else arch
                exec {
                    commandLine("bash", "build.sh", libs.versions.llamacpp.get(), arch, cp)
                }
            } else
                exec {
                    commandLine("${project.projectDir}/build.cmd", libs.versions.llamacpp.get(), cp)
                }

            // for ci to upload to S3
            val ciDir = project.projectDir / "jnilib/${libs.versions.djl.get()}/"
            copy {
                from(project.projectDir / "build/jnilib")
                into(ciDir)
            }
            delete("$home/.djl.ai/llama")
        }
    }

    clean {
        doFirst {
            delete("$home/.djl.ai/llama")
        }
    }
}