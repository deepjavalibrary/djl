plugins {
    ai.djl.javaProject
    ai.djl.publish
    signing
}

group = "ai.djl.tensorflow"

val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = libs.versions.tensorflow.get() + if (isRelease) "" else "-SNAPSHOT"

val download by configurations.registering

dependencies {
    download("org.tensorflow:tensorflow-core-api:0.5.0:linux-x86_64-gpu")
    download("org.tensorflow:tensorflow-core-api:0.5.0:linux-x86_64")
    download("org.tensorflow:tensorflow-core-api:0.5.0:macosx-x86_64")
    download("org.tensorflow:tensorflow-core-api:0.5.0:windows-x86_64")
    download("org.tensorflow:tensorflow-core-api:0.5.0:windows-x86_64-gpu")
}

tasks {
    register("uploadTensorflowNativeLibs") {
        doLast {
            delete(buildDirectory / "download")
            delete(buildDirectory / "native")

            copy {
                from(download)
                into(buildDirectory / "download")
            }

            fileTree(buildDirectory / "download").forEach { f ->
                copy {
                    from(zipTree(f)) {
                        exclude(
                            "**/pom.xml",
                            "**/*.properties",
                            "**/*.h",
                            "**/*.hpp",
                            "**/*.cmake",
                            "META-INF/**"
                        )
                    }
                    into(buildDirectory / "native")
                    includeEmptyDirs = false
                }
            }

            exec {
                commandLine("sh", "-c", "find $buildDirectory/native -type f | xargs gzip")
            }

            val tfUnzipDir = buildDirectory / "native/org/tensorflow/internal/c_api"

            ant.withGroovyBuilder {
                "move"("file" to "$tfUnzipDir/linux-x86_64/", "toFile" to "$buildDirectory/native/linux/cpu/")
                "move"("file" to "$tfUnzipDir/linux-x86_64-gpu/", "toFile" to "$buildDirectory/native/linux/cu113/")
                "move"("file" to "$tfUnzipDir/macosx-x86_64/", "toFile" to "$buildDirectory/native/osx/cpu/")
                "move"("file" to "$tfUnzipDir/windows-x86_64/", "toFile" to "$buildDirectory/native/win/cpu/")
                "move"("file" to "$tfUnzipDir/windows-x86_64-gpu/", "toFile" to "$buildDirectory/native/win/cu113/")
            }

            (buildDirectory / "native/files.txt").text = buildString {
                val uploadDirs = listOf(
                    buildDirectory / "native/linux/cpu/",
                    buildDirectory / "native/linux/cu113/",
                    buildDirectory / "native/osx/cpu/",
                    buildDirectory / "native/win/cpu/",
                    buildDirectory / "native/win/cu113/"
                )
                for (item in uploadDirs)
                    fileTree(item).files.map { it.name }.forEach {
                        val out = item.relativeTo(buildDirectory / "native/").absolutePath
                        appendLine(out + it)
                    }
            }
            delete(
                buildDirectory / "native/org/",
                buildDirectory / "native/com/",
                buildDirectory / "native/google/",
                buildDirectory / "native/module-info.class.gz"
            )

            exec {
                commandLine(
                    "aws",
                    "s3",
                    "sync",
                    "$buildDirectory/native/",
                    "s3://djl-ai/publish/tensorflow-${libs.versions.tensorflow.get()}/"
                )
            }
        }
    }

    jar {
        // this line is to enforce gradle to build the jar
        // otherwise it don't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")
        doFirst {
            val dir = buildDirectory / "classes/java/main/native/lib"
            dir.mkdirs()
            val propFile = dir / "tensorflow.properties"
            var versionName = project.version.toString()
            if (!isRelease)
                versionName += "-$nowFormatted"
            propFile.text = "placeholder=true\nversion=${versionName}\n"
        }
    }

    withType<GenerateModuleMetadata> { enabled = false }
}

java {
    withJavadocJar()
    withSourcesJar()
}

signing {
    isRequired = project.hasProperty("staging") || project.hasProperty("snapshot")
    val signingKey = findProperty("signingKey").toString()
    val signingPassword = findProperty("signingPassword").toString()
    useInMemoryPgpKeys(signingKey, signingPassword)
    sign(publishing.publications["maven"])
}

val BINARY_ROOT = buildDirectory / "download"
val flavorNames = BINARY_ROOT.list() ?: emptyArray()
for (flavor in flavorNames) {

    val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()

    val artifactsNames = ArrayList<Task>()

    for (osName in platformNames) {
        tasks.create<Jar>("$flavor-${osName}Jar") {
            doFirst {
                val propFile = BINARY_ROOT / flavor / osName / "native/lib/tensorflow.properties"
                propFile.delete()
                val dsStore = BINARY_ROOT / flavor / osName / "native/lib/.DS_Store"
                dsStore.delete()

                val versionName = "${project.version}-$nowFormatted"
                val dir = BINARY_ROOT / flavor / osName / "native/lib"
                propFile.text = buildString {
                    append("version=$versionName\nclassifier=$flavor-$osName-x86_64\nlibraries=")
                    var first = true
                    for (name in dir.list()!!.sorted()) {
                        if (first)
                            first = false
                        else
                            append(',')
                        append(name)
                    }
                }

                from("src/main/resources")
            }
            from(BINARY_ROOT / flavor / osName)
            archiveClassifier = "$osName-x86_64"
            archiveBaseName = "tensorflow-native-$flavor"

            manifest {
                attributes("Automatic-Module-Name" to "ai.djl.tensorflow_native_${flavor}_$osName")
            }
        }
        artifactsNames.add(tasks["$flavor-${osName}Jar"])
    }

    // Only publish if the project directory equals the current directory
    // This means that publishing from the main project does not publish the native jars
    // and the native jars have to be published separately
    if (project.projectDir.toString() == System.getProperty("user.dir")) {
        publishing.publications.create<MavenPublication>(flavor) {
            artifactId = "tensorflow-native-$flavor"
            from(components["java"])
            setArtifacts(artifactsNames)
            artifact(tasks.jar)
            artifact(tasks.javadocJar)
            artifact(tasks.sourcesJar)
            pom {
                name = "DJL release for TensorFlow native binaries"
                description = "Deep Java Library (DJL) provided TensorFlow native library binary distribution"
                url = "http://www.djl.ai/engines/tensorflow/${project.name}"
                packaging = "jar"

                licenses {
                    license {
                        name = "The Apache License, Version 2.0"
                        url = "https://www.apache.org/licenses/LICENSE-2.0"
                    }
                }

                scm {
                    connection = "scm:git:git@github.com:deepjavalibrary/djl.git"
                    developerConnection = "scm:git:git@github.com:deepjavalibrary/djl.git"
                    url = "https://github.com/deepjavalibrary/djl"
                    tag = "HEAD"
                }

                developers {
                    developer {
                        name = "DJL.AI Team"
                        email = "djl-dev@amazon.com"
                        organization = "Amazon AI"
                        organizationUrl = "https://amazon.com"
                    }
                }
            }
        }
    }
}

tasks {
    // Gradle 8.0 requires explicitly dependency
    withType<PublishToMavenRepository> {
        for (flavor in flavorNames) {
            dependsOn("sign${flavor.substring(0, 1).uppercase() + flavor.substring(1)}Publication")

            val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }
    withType<Sign> {
        for (flavor in flavorNames) {
            val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }
}

publishing.repositories {
    maven {
        if (project.hasProperty("snapshot")) {
            name = "snapshot"
            url = uri("https://oss.sonatype.org/content/repositories/snapshots/")
            credentials {
                username = findProperty("ossrhUsername").toString()
                password = findProperty("ossrhPassword").toString()
            }
        } else if (project.hasProperty("staging")) {
            name = "staging"
            url = uri("https://oss.sonatype.org/service/local/staging/deploy/maven2/")
            credentials {
                username = findProperty("ossrhUsername").toString()
                password = findProperty("ossrhPassword").toString()
            }
        } else {
            name = "local"
            url = uri("build/repo")
        }
    }
}

tasks.register("downloadTensorflowNativeLib") {
    doLast {
        val url = "https://publish.djl.ai/tensorflow-${libs.versions.tensorflow.get()}"
        // @formatter:off
        val files = mapOf(
                "linux/cpu/libjnitensorflow.so.gz"                           to "cpu/linux/native/lib/libjnitensorflow.so",
                "linux/cpu/libtensorflow_cc.so.2.gz"                         to "cpu/linux/native/lib/libtensorflow_cc.so.2",
                "linux/cpu/libtensorflow_framework.so.2.gz"                  to "cpu/linux/native/lib/libtensorflow_framework.so.2",
                "linux/cpu/LICENSE.gz"                                       to "cpu/linux/META-INF/LICENSE",
                "linux/cpu/THIRD_PARTY_TF_JNI_LICENSES.gz"                   to "cpu/linux/META-INF/THIRD_PARTY_TF_JNI_LICENSES",
                "linux/cu113/libjnitensorflow.so.gz"                         to "cu113/linux/native/lib/libjnitensorflow.so",
                "linux/cu113/libtensorflow_cc.so.2.gz"                       to "cu113/linux/native/lib/libtensorflow_cc.so.2",
                "linux/cu113/libtensorflow_framework.so.2.gz"                to "cu113/linux/native/lib/libtensorflow_framework.so.2",
                "linux/cu113/LICENSE.gz"                                     to "cu113/linux/META-INF/LICENSE",
                "linux/cu113/THIRD_PARTY_TF_JNI_LICENSES.gz"                 to "cu113/linux/META-INF/THIRD_PARTY_TF_JNI_LICENSES",
                "osx/cpu/libjnitensorflow.dylib.gz"                          to "cpu/osx/native/lib/libjnitensorflow.dylib",
                "osx/cpu/libtensorflow_cc.2.dylib.gz"                        to "cpu/osx/native/lib/libtensorflow_cc.2.dylib",
                "osx/cpu/libtensorflow_framework.2.dylib.gz"                 to "cpu/osx/native/lib/libtensorflow_framework.2.dylib",
                "osx/cpu/LICENSE.gz"                                         to "cpu/osx/META-INF/LICENSE",
                "osx/cpu/THIRD_PARTY_TF_JNI_LICENSES.gz"                     to "cpu/osx/META-INF/THIRD_PARTY_TF_JNI_LICENSES",
                "win/cpu/api-ms-win-core-console-l1-1-0.dll.gz"              to "cpu/win/native/lib/api-ms-win-core-console-l1-1-0.dll",
                "win/cpu/api-ms-win-core-datetime-l1-1-0.dll.gz"             to "cpu/win/native/lib/api-ms-win-core-datetime-l1-1-0.dll",
                "win/cpu/api-ms-win-core-debug-l1-1-0.dll.gz"                to "cpu/win/native/lib/api-ms-win-core-debug-l1-1-0.dll",
                "win/cpu/api-ms-win-core-errorhandling-l1-1-0.dll.gz"        to "cpu/win/native/lib/api-ms-win-core-errorhandling-l1-1-0.dll",
                "win/cpu/api-ms-win-core-file-l1-1-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-core-file-l1-1-0.dll",
                "win/cpu/api-ms-win-core-file-l1-2-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-core-file-l1-2-0.dll",
                "win/cpu/api-ms-win-core-file-l2-1-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-core-file-l2-1-0.dll",
                "win/cpu/api-ms-win-core-handle-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-core-handle-l1-1-0.dll",
                "win/cpu/api-ms-win-core-heap-l1-1-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-core-heap-l1-1-0.dll",
                "win/cpu/api-ms-win-core-interlocked-l1-1-0.dll.gz"          to "cpu/win/native/lib/api-ms-win-core-interlocked-l1-1-0.dll",
                "win/cpu/api-ms-win-core-libraryloader-l1-1-0.dll.gz"        to "cpu/win/native/lib/api-ms-win-core-libraryloader-l1-1-0.dll",
                "win/cpu/api-ms-win-core-localization-l1-2-0.dll.gz"         to "cpu/win/native/lib/api-ms-win-core-localization-l1-2-0.dll",
                "win/cpu/api-ms-win-core-memory-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-core-memory-l1-1-0.dll",
                "win/cpu/api-ms-win-core-namedpipe-l1-1-0.dll.gz"            to "cpu/win/native/lib/api-ms-win-core-namedpipe-l1-1-0.dll",
                "win/cpu/api-ms-win-core-processenvironment-l1-1-0.dll.gz"   to "cpu/win/native/lib/api-ms-win-core-processenvironment-l1-1-0.dll",
                "win/cpu/api-ms-win-core-processthreads-l1-1-0.dll.gz"       to "cpu/win/native/lib/api-ms-win-core-processthreads-l1-1-0.dll",
                "win/cpu/api-ms-win-core-processthreads-l1-1-1.dll.gz"       to "cpu/win/native/lib/api-ms-win-core-processthreads-l1-1-1.dll",
                "win/cpu/api-ms-win-core-profile-l1-1-0.dll.gz"              to "cpu/win/native/lib/api-ms-win-core-profile-l1-1-0.dll",
                "win/cpu/api-ms-win-core-rtlsupport-l1-1-0.dll.gz"           to "cpu/win/native/lib/api-ms-win-core-rtlsupport-l1-1-0.dll",
                "win/cpu/api-ms-win-core-string-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-core-string-l1-1-0.dll",
                "win/cpu/api-ms-win-core-synch-l1-1-0.dll.gz"                to "cpu/win/native/lib/api-ms-win-core-synch-l1-1-0.dll",
                "win/cpu/api-ms-win-core-synch-l1-2-0.dll.gz"                to "cpu/win/native/lib/api-ms-win-core-synch-l1-2-0.dll",
                "win/cpu/api-ms-win-core-sysinfo-l1-1-0.dll.gz"              to "cpu/win/native/lib/api-ms-win-core-sysinfo-l1-1-0.dll",
                "win/cpu/api-ms-win-core-timezone-l1-1-0.dll.gz"             to "cpu/win/native/lib/api-ms-win-core-timezone-l1-1-0.dll",
                "win/cpu/api-ms-win-core-util-l1-1-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-core-util-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-convert-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-crt-convert-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-environment-l1-1-0.dll.gz"           to "cpu/win/native/lib/api-ms-win-crt-environment-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-filesystem-l1-1-0.dll.gz"            to "cpu/win/native/lib/api-ms-win-crt-filesystem-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-heap-l1-1-0.dll.gz"                  to "cpu/win/native/lib/api-ms-win-crt-heap-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-locale-l1-1-0.dll.gz"                to "cpu/win/native/lib/api-ms-win-crt-locale-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-math-l1-1-0.dll.gz"                  to "cpu/win/native/lib/api-ms-win-crt-math-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-multibyte-l1-1-0.dll.gz"             to "cpu/win/native/lib/api-ms-win-crt-multibyte-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-runtime-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-crt-runtime-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-stdio-l1-1-0.dll.gz"                 to "cpu/win/native/lib/api-ms-win-crt-stdio-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-string-l1-1-0.dll.gz"                to "cpu/win/native/lib/api-ms-win-crt-string-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-time-l1-1-0.dll.gz"                  to "cpu/win/native/lib/api-ms-win-crt-time-l1-1-0.dll",
                "win/cpu/api-ms-win-crt-utility-l1-1-0.dll.gz"               to "cpu/win/native/lib/api-ms-win-crt-utility-l1-1-0.dll",
                "win/cpu/concrt140.dll.gz"                                   to "cpu/win/native/lib/concrt140.dll",
                "win/cpu/jnitensorflow.dll.gz"                               to "cpu/win/native/lib/jnitensorflow.dll",
                "win/cpu/libiomp5md.dll.gz"                                  to "cpu/win/native/lib/libiomp5md.dll",
                "win/cpu/LICENSE.gz"                                         to "cpu/win/META-INF/LICENSE",
                "win/cpu/msvcp140.dll.gz"                                    to "cpu/win/native/lib/msvcp140.dll",
                "win/cpu/tensorflow_cc.dll.gz"                               to "cpu/win/native/lib/tensorflow_cc.dll",
                "win/cpu/THIRD_PARTY_TF_JNI_LICENSES.gz"                     to "cpu/win/META-INF/THIRD_PARTY_TF_JNI_LICENSES",
                "win/cpu/ucrtbase.dll.gz"                                    to "cpu/win/native/lib/ucrtbase.dll",
                "win/cpu/vcomp140.dll.gz"                                    to "cpu/win/native/lib/vcomp140.dll",
                "win/cpu/vcruntime140_1.dll.gz"                              to "cpu/win/native/lib/vcruntime140_1.dll",
                "win/cpu/vcruntime140.dll.gz"                                to "cpu/win/native/lib/vcruntime140.dll",
                "win/cu113/api-ms-win-core-console-l1-1-0.dll.gz"            to "cu113/win/native/lib/api-ms-win-core-console-l1-1-0.dll",
                "win/cu113/api-ms-win-core-datetime-l1-1-0.dll.gz"           to "cu113/win/native/lib/api-ms-win-core-datetime-l1-1-0.dll",
                "win/cu113/api-ms-win-core-debug-l1-1-0.dll.gz"              to "cu113/win/native/lib/api-ms-win-core-debug-l1-1-0.dll",
                "win/cu113/api-ms-win-core-errorhandling-l1-1-0.dll.gz"      to "cu113/win/native/lib/api-ms-win-core-errorhandling-l1-1-0.dll",
                "win/cu113/api-ms-win-core-file-l1-1-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-core-file-l1-1-0.dll",
                "win/cu113/api-ms-win-core-file-l1-2-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-core-file-l1-2-0.dll",
                "win/cu113/api-ms-win-core-file-l2-1-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-core-file-l2-1-0.dll",
                "win/cu113/api-ms-win-core-handle-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-core-handle-l1-1-0.dll",
                "win/cu113/api-ms-win-core-heap-l1-1-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-core-heap-l1-1-0.dll",
                "win/cu113/api-ms-win-core-interlocked-l1-1-0.dll.gz"        to "cu113/win/native/lib/api-ms-win-core-interlocked-l1-1-0.dll",
                "win/cu113/api-ms-win-core-libraryloader-l1-1-0.dll.gz"      to "cu113/win/native/lib/api-ms-win-core-libraryloader-l1-1-0.dll",
                "win/cu113/api-ms-win-core-localization-l1-2-0.dll.gz"       to "cu113/win/native/lib/api-ms-win-core-localization-l1-2-0.dll",
                "win/cu113/api-ms-win-core-memory-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-core-memory-l1-1-0.dll",
                "win/cu113/api-ms-win-core-namedpipe-l1-1-0.dll.gz"          to "cu113/win/native/lib/api-ms-win-core-namedpipe-l1-1-0.dll",
                "win/cu113/api-ms-win-core-processenvironment-l1-1-0.dll.gz" to "cu113/win/native/lib/api-ms-win-core-processenvironment-l1-1-0.dll",
                "win/cu113/api-ms-win-core-processthreads-l1-1-0.dll.gz"     to "cu113/win/native/lib/api-ms-win-core-processthreads-l1-1-0.dll",
                "win/cu113/api-ms-win-core-processthreads-l1-1-1.dll.gz"     to "cu113/win/native/lib/api-ms-win-core-processthreads-l1-1-1.dll",
                "win/cu113/api-ms-win-core-profile-l1-1-0.dll.gz"            to "cu113/win/native/lib/api-ms-win-core-profile-l1-1-0.dll",
                "win/cu113/api-ms-win-core-rtlsupport-l1-1-0.dll.gz"         to "cu113/win/native/lib/api-ms-win-core-rtlsupport-l1-1-0.dll",
                "win/cu113/api-ms-win-core-string-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-core-string-l1-1-0.dll",
                "win/cu113/api-ms-win-core-synch-l1-1-0.dll.gz"              to "cu113/win/native/lib/api-ms-win-core-synch-l1-1-0.dll",
                "win/cu113/api-ms-win-core-synch-l1-2-0.dll.gz"              to "cu113/win/native/lib/api-ms-win-core-synch-l1-2-0.dll",
                "win/cu113/api-ms-win-core-sysinfo-l1-1-0.dll.gz"            to "cu113/win/native/lib/api-ms-win-core-sysinfo-l1-1-0.dll",
                "win/cu113/api-ms-win-core-timezone-l1-1-0.dll.gz"           to "cu113/win/native/lib/api-ms-win-core-timezone-l1-1-0.dll",
                "win/cu113/api-ms-win-core-util-l1-1-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-core-util-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-convert-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-crt-convert-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-environment-l1-1-0.dll.gz"         to "cu113/win/native/lib/api-ms-win-crt-environment-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-filesystem-l1-1-0.dll.gz"          to "cu113/win/native/lib/api-ms-win-crt-filesystem-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-heap-l1-1-0.dll.gz"                to "cu113/win/native/lib/api-ms-win-crt-heap-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-locale-l1-1-0.dll.gz"              to "cu113/win/native/lib/api-ms-win-crt-locale-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-math-l1-1-0.dll.gz"                to "cu113/win/native/lib/api-ms-win-crt-math-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-multibyte-l1-1-0.dll.gz"           to "cu113/win/native/lib/api-ms-win-crt-multibyte-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-runtime-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-crt-runtime-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-stdio-l1-1-0.dll.gz"               to "cu113/win/native/lib/api-ms-win-crt-stdio-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-string-l1-1-0.dll.gz"              to "cu113/win/native/lib/api-ms-win-crt-string-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-time-l1-1-0.dll.gz"                to "cu113/win/native/lib/api-ms-win-crt-time-l1-1-0.dll",
                "win/cu113/api-ms-win-crt-utility-l1-1-0.dll.gz"             to "cu113/win/native/lib/api-ms-win-crt-utility-l1-1-0.dll",
                "win/cu113/concrt140.dll.gz"                                 to "cu113/win/native/lib/concrt140.dll",
                "win/cu113/jnitensorflow.dll.gz"                             to "cu113/win/native/lib/jnitensorflow.dll",
                "win/cu113/libiomp5md.dll.gz"                                to "cu113/win/native/lib/libiomp5md.dll",
                "win/cu113/LICENSE.gz"                                       to "cu113/win/META-INF/LICENSE",
                "win/cu113/msvcp140.dll.gz"                                  to "cu113/win/native/lib/msvcp140.dll",
                "win/cu113/tensorflow_cc.dll.gz"                             to "cu113/win/native/lib/tensorflow_cc.dll",
                "win/cu113/THIRD_PARTY_TF_JNI_LICENSES.gz"                   to "cu113/win/META-INF/THIRD_PARTY_TF_JNI_LICENSES",
                "win/cu113/ucrtbase.dll.gz"                                  to "cu113/win/native/lib/ucrtbase.dll",
                "win/cu113/vcomp140.dll.gz"                                  to "cu113/win/native/lib/vcomp140.dll",
                "win/cu113/vcruntime140_1.dll.gz"                            to "cu113/win/native/lib/vcruntime140_1.dll",
                "win/cu113/vcruntime140.dll.gz"                              to "cu113/win/native/lib/vcruntime140.dll")
        // @formatter:on
        for ((key, value) in files) {
            project.logger.lifecycle("Downloading $url/$key")
            val file = BINARY_ROOT / value
            file.parentFile.mkdirs()
            "$url/$key".url gzipInto file
        }
    }
}
