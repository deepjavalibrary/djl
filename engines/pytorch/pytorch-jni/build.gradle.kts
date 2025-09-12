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

        val djlVersion = libs.versions.djl.get()
        val logger = project.logger
        val dir = project.projectDir
        val buildDir = buildDirectory
        val nativeDir = project.parent!!.projectDir / "pytorch-native/jnilib/${djlVersion}/"
        val version = project.version
        val hasJni = project.hasProperty("jni")
        val ptVer = ptVersion
        val injected = project.objects.newInstance<InjectedOps>()

        doFirst {
            val url = "https://publish.djl.ai/pytorch/$ptVer/jnilib/${djlVersion}"
            val files = listOf(
                "linux-x86_64/cpu/libdjl_torch.so",
                "osx-aarch64/cpu/libdjl_torch.dylib",
                "win-x86_64/cpu/djl_torch.dll"
            ) + when {
                ptVer.matches(Regex("2.7.\\d")) -> listOf(
                    "linux-aarch64/cpu/libdjl_torch.so",
                    "linux-x86_64/cu128/libdjl_torch.so",
                    "win-x86_64/cu128/djl_torch.dll"
                )

                ptVer.matches(Regex("2.[4-5].\\d")) -> listOf(
                    "linux-x86_64/cpu-precxx11/libdjl_torch.so",
                    "linux-aarch64/cpu-precxx11/libdjl_torch.so",
                    "linux-x86_64/cu124/libdjl_torch.so",
                    "linux-x86_64/cu124-precxx11/libdjl_torch.so",
                    "win-x86_64/cu124/djl_torch.dll"
                )

                ptVer.matches(Regex("2.[1-3].\\d")) -> listOf(
                    "linux-x86_64/cpu-precxx11/libdjl_torch.so",
                    "linux-aarch64/cpu-precxx11/libdjl_torch.so",
                    "linux-x86_64/cu121/libdjl_torch.so",
                    "linux-x86_64/cu121-precxx11/libdjl_torch.so",
                    "win-x86_64/cu121/djl_torch.dll",
                )

                ptVer.startsWith("1.13.") -> listOf(
                    "linux-x86_64/cpu-precxx11/libdjl_torch.so",
                    "linux-aarch64/cpu-precxx11/libdjl_torch.so",
                    "linux-x86_64/cu117/libdjl_torch.so",
                    "win-x86_64/cu117/djl_torch.dll",
                )

                else -> throw GradleException("Unsupported version: $ptVer.")
            }
            val jnilibDir = dir / "jnilib" / djlVersion
            for (entry in files) {
                val file = jnilibDir / entry
                if (file.exists())
                    logger.lifecycle("prebuilt or cached file found for $entry")
                else {
                    val jnilibFile = nativeDir / entry
                    if (jnilibFile.exists()) {
                        logger.lifecycle("Copying $jnilibFile")
                        injected.fs.copy {
                            from(jnilibFile)
                            into(file.parent)
                        }
                    } else if (!hasJni) {
                        logger.lifecycle("Downloading $url/$entry")
                        file.parentFile.mkdirs()
                        "$url/$entry".url into file
                    }
                }
            }
            injected.fs.copy {
                from(jnilibDir)
                into(buildDir / "classes/java/main/jnilib")
            }

            // write properties
            val propFile = buildDir / "classes/java/main/jnilib/pytorch.properties"
            propFile.text = "jni_version=$version"
        }
    }

    clean {
        val injected = project.objects.newInstance<InjectedOps>()
        val files = fileTree("$home/.djl.ai/pytorch/") {
            include("**/*djl_torch.*")
        }
        doFirst {
            injected.fs.delete {
                delete("jnilib")
                delete(files)
            }
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

interface InjectedOps {
    @get:Inject
    val fs: FileSystemOperations
}
