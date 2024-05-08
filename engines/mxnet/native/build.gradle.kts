import java.net.URL

plugins {
    ai.djl.javaProject
    `maven-publish`
    signing
}

group = "ai.djl.mxnet"

val VERSION = libs.versions.mxnet.get()
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = VERSION + if (isRelease) "" else "-SNAPSHOT"

tasks {
    // Create a placeholder jar without classifier to pass sonatype tests but throws an Exception if loaded
    jar {
        val placeholder = layout.buildDirectory / "placeholder"
        // this line is to enforce gradle to build the jar otherwise it doesn't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")
        doFirst {
            var versionName = project.version.toString()
            if (!isRelease)
                versionName += "-$nowFormatted"
            val dir = placeholder / "native/lib"
            dir.mkdirs()
            val propFile = placeholder / "native/lib/mxnet.properties"
            propFile.text = "placeholder=true\nversion=$versionName\n"
        }
        from(placeholder)
    }

    withType<GenerateModuleMetadata> { enabled = false }

    signing {
        isRequired = project.hasProperty("staging") || project.hasProperty("snapshot")
        val signingKey = findProperty("signingKey").toString()
        val signingPassword = findProperty("signingPassword").toString()
        useInMemoryPgpKeys(signingKey, signingPassword)
        sign(publishing.publications)
    }

    val BINARY_ROOT = layout.buildDirectory / "download"
    val flavorNames = file(BINARY_ROOT).list() ?: emptyArray()
    for (flavor in flavorNames) {

        val platformNames = (BINARY_ROOT / flavor).list() ?: emptyArray()

        val artifactsNames = ArrayList<Task>()

        for (osName in platformNames) {
            register<Jar>("$flavor-${osName}Jar") {
                doFirst {
                    val propFile = BINARY_ROOT / flavor / osName / "native/lib/mxnet.properties"
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
                    val metaInf = BINARY_ROOT / flavor / osName / "META-INF"
                    metaInf.mkdirs()
                    val licenseFile = metaInf / "LICENSE"
                    licenseFile.text = URL("https://raw.githubusercontent.com/apache/incubator-mxnet/master/LICENSE").text

                    val binaryLicenseFile = metaInf / "LICENSE.binary.dependencies"
                    binaryLicenseFile.text = URL("https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/dependencies/LICENSE.binary.dependencies").text

                    from("src/main/resources")
                }
                from(BINARY_ROOT / flavor / osName)
                archiveClassifier = "$osName-x86_64"
                archiveBaseName = "mxnet-native-$flavor"

                manifest {
                    attributes("Automatic-Module-Name" to "ai.djl.mxnet_native_${flavor}_$osName")
                }
            }
            artifactsNames += named("${flavor}-${osName}Jar").get()
        }

        // Only publish if the project directory equals the current directory
        // This means that publishing from the main project does not publish the native jars
        // and the native jars have to be published separately
        if (project.projectDir.toString() == System.getProperty("user.dir")) {
            publishing.publications.register<MavenPublication>(flavor) {
                artifactId = "mxnet-native-$flavor"
                from(project.components["java"])
                setArtifacts(artifactsNames)
                artifact(jar)
                artifact(named("javadocJar"))
                artifact(named("sourcesJar"))
                pom {
                    name = "DJL release for Apache MXNet native binaries"
                    description = "Deep Java Library (DJL) provided Apache MXNet native library binary distribution"
                    url = "http://www.djl.ai/engines/mxnet/native"
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

    register("downloadMxnetNativeLib") {
        doLast {
            val url = "https://publish.djl.ai/mxnet-$VERSION"
            // @formatter:off
            val files = mapOf("linux/common/libgfortran.so.3.gz" to "mkl/linux/native/lib/libgfortran.so.3",
                              "linux/common/libgomp.so.1.gz"     to "mkl/linux/native/lib/libgomp.so.1",
                              "linux/common/libopenblas.so.0.gz" to "mkl/linux/native/lib/libopenblas.so.0",
                              "linux/common/libquadmath.so.0.gz" to "mkl/linux/native/lib/libquadmath.so.0",
                              "linux/mkl/libmxnet.so.gz"         to "mkl/linux/native/lib/libmxnet.so",
                              "linux/cu102mkl/libmxnet.so.gz"    to "cu102mkl/linux/native/lib/libmxnet.so",
                              "linux/cu112mkl/libmxnet.so.gz"    to "cu112mkl/linux/native/lib/libmxnet.so",
                              "osx/mkl/libmxnet.dylib.gz"        to "mkl/osx/native/lib/libmxnet.dylib",
                              "win/common/libgcc_s_seh-1.dll.gz" to "mkl/win/native/lib/libgcc_s_seh-1.dll",
                              "win/common/libgfortran-3.dll.gz"  to "mkl/win/native/lib/libgfortran-3.dll",
                              "win/common/libopenblas.dll.gz"    to "mkl/win/native/lib/libopenblas.dll",
                              "win/common/libquadmath-0.dll.gz"  to "mkl/win/native/lib/libquadmath-0.dll",
                              "win/mkl/libmxnet.dll.gz"          to "mkl/win/native/lib/mxnet.dll")
            // @formatter:on
            for ((key, value) in files) {
                project.logger.lifecycle("Downloading $url/$key")
                val file = BINARY_ROOT / value
                file.parentFile.mkdirs()
                URL("$url/$key") gzipInto file
            }

            copy {
                from(BINARY_ROOT / "mkl/linux/native/lib") {
                    exclude("**/libmxnet.so")
                }
                into(BINARY_ROOT / "cu102mkl/linux/native/lib")
            }
            copy {
                from(BINARY_ROOT / "mkl/linux/native/lib") {
                    exclude("**/libmxnet.so")
                }
                into(BINARY_ROOT / "cu112mkl/linux/native/lib")
            }
        }
    }
}

java {
    withJavadocJar()
    withSourcesJar()
}