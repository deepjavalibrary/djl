import java.net.URL

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

val ptVersion: String = when {
    project.hasProperty("pt_version") && project.property("pt_version") != "" ->
        project.property("pt_version").toString()
    else -> libs.versions.pytorch.get()
}

group = "ai.djl.pytorch"
version = ptVersion + '-' + libs.versions.djl.get()
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
if (!isRelease)
    version = ptVersion + "-${libs.versions.djl.get()}-SNAPSHOT"

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.dir(buildDirectory / "classes/java/main/jnilib")
        doFirst {
            val url = "https://publish.djl.ai/pytorch/$ptVersion/jnilib/${libs.versions.djl.get()}"
            val files = listOf("linux-x86_64/cpu/libdjl_torch.so",
                               "linux-x86_64/cpu-precxx11/libdjl_torch.so",
                               "linux-aarch64/cpu-precxx11/libdjl_torch.so",
                               "osx-aarch64/cpu/libdjl_torch.dylib",
                               "win-x86_64/cpu/djl_torch.dll") + when {
                            ptVersion.startsWith("2.3.") -> listOf(
                                "linux-x86_64/cu121/libdjl_torch.so",
                                "linux-x86_64/cu121-precxx11/libdjl_torch.so",
                                "win-x86_64/cu121/djl_torch.dll")
                            ptVersion.startsWith("2.1.") || ptVersion.startsWith("2.2.") -> listOf(
                                "linux-x86_64/cu121/libdjl_torch.so",
                                "linux-x86_64/cu121-precxx11/libdjl_torch.so",
                                "win-x86_64/cu121/djl_torch.dll",
                                "osx-x86_64/cpu/libdjl_torch.dylib")
                            ptVersion.startsWith("2.0.") -> listOf(
                                "linux-x86_64/cu118/libdjl_torch.so",
                                "linux-x86_64/cu118-precxx11/libdjl_torch.so",
                                "win-x86_64/cu118/djl_torch.dll",
                                "osx-x86_64/cpu/libdjl_torch.dylib")
                            ptVersion.startsWith("1.13.") -> listOf(
                                "linux-x86_64/cu117/libdjl_torch.so",
                                "linux-x86_64/cu117-precxx11/libdjl_torch.so",
                                "win-x86_64/cu117/djl_torch.dll",
                                "osx-x86_64/cpu/libdjl_torch.dylib")
                            else -> throw GradleException("Unsupported version: $ptVersion.")
                        }
            val jnilibDir = project.projectDir / "jnilib" / libs.versions.djl.get()
            for (entry in files) {
                val file = jnilibDir / entry
                if (file.exists())
                    project.logger.lifecycle("prebuilt or cached file found for $entry")
                else {
                    val nativeDir = project.parent!!.projectDir / "pytorch-native/jnilib/${libs.versions.djl.get()}/"
                    val jnilibFile = nativeDir / entry
                    if (jnilibFile.exists()) {
                        project.logger.lifecycle("Copying $jnilibFile")
                        copy {
                            from(jnilibFile)
                            into(file.parent)
                        }
                    } else if (!project.hasProperty("jni")) {
                        project.logger.lifecycle("Downloading $url/$entry")
                        file.parentFile.mkdirs()
                        "$url/$entry".url into file
                    }
                }
            }
            copy {
                from(jnilibDir)
                into(buildDirectory / "classes/java/main/jnilib")
            }

            // write properties
            val propFile = buildDirectory / "classes/java/main/jnilib/pytorch.properties"
            propFile.text = "jni_version=" + project.version
        }
    }

    clean {
        doFirst {
            delete("jnilib")
            delete(fileTree("$home/.djl.ai/pytorch/") {
                include("**/*djl_torch.*")
            })
        }
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for PyTorch"
                description = "Deep Java Library (DJL) Engine Adapter for PyTorch"
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}