import java.net.HttpURLConnection
import java.net.URL
import java.util.Base64
import kotlin.collections.ArrayList

plugins {
    ai.djl.javaProject
    `maven-publish`
    signing
}

group = "ai.djl.mxnet"

val mxnetVersion = libs.versions.mxnet.get()
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = mxnetVersion + if (isRelease) "" else "-SNAPSHOT"

tasks {
    // Create a placeholder jar without classifier to pass sonatype tests but throws an Exception if loaded
    jar {
        val placeholder = buildDirectory / "placeholder"
        // this line is to enforce gradle to build the jar otherwise it doesn't generate the placeholder jar at times
        // when there is no java code inside src/main
        outputs.dir("build/libs")

        var versionName = project.version.toString()
        if (!isRelease)
            versionName += "-$nowFormatted"

        doFirst {
            val dir = placeholder / "native/lib"
            dir.mkdirs()
            val propFile = placeholder / "native/lib/mxnet.properties"
            propFile.text = "placeholder=true\nversion=$versionName\n"
        }
        from(placeholder)
    }

    java {
        withJavadocJar()
        withSourcesJar()
    }

    withType<GenerateModuleMetadata> { enabled = false }

    val binaryRoot = buildDirectory / "download"
    val requireSigning = project.hasProperty("staging") || project.hasProperty("snapshot")
    val flavorNames = file(binaryRoot).list() ?: emptyArray()
    for (flavor in flavorNames) {

        val platformNames = (binaryRoot / flavor).list() ?: emptyArray()

        val artifactsNames = ArrayList<Task>()

        for (osName in platformNames) {
            register<Jar>("$flavor-${osName}Jar") {
                doFirst {
                    val propFile = binaryRoot / flavor / osName / "native/lib/mxnet.properties"
                    propFile.delete()
                    val dsStore = binaryRoot / flavor / osName / "native/lib/.DS_Store"
                    dsStore.delete()

                    val versionName = "${project.version}-$nowFormatted"
                    val dir = binaryRoot / flavor / osName / "native/lib"
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
                    val metaInf = binaryRoot / flavor / osName / "META-INF"
                    metaInf.mkdirs()
                    val licenseFile = metaInf / "LICENSE"
                    licenseFile.text =
                        "https://raw.githubusercontent.com/apache/incubator-mxnet/master/LICENSE".url.text

                    val binaryLicenseFile = metaInf / "LICENSE.binary.dependencies"
                    binaryLicenseFile.text =
                        "https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/dependencies/LICENSE.binary.dependencies".url.text

                    from("src/main/resources")
                }
                from(binaryRoot / flavor / osName)
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

    signing {
        isRequired = requireSigning
        if (requireSigning) {
            val signingKey = findProperty("signingKey").toString()
            val signingPassword = findProperty("signingPassword").toString()
            useInMemoryPgpKeys(signingKey, signingPassword)
            sign(publishing.publications)
        }
    }

    // Gradle 8.0 requires explicitly dependency
    withType<PublishToMavenRepository> {
        for (flavor in flavorNames) {
            if (requireSigning) {
                dependsOn("sign${flavor.take(1).uppercase() + flavor.substring(1)}Publication")
            }

            val platformNames = (binaryRoot / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }
    withType<Sign> {
        for (flavor in flavorNames) {
            val platformNames = (binaryRoot / flavor).list() ?: emptyArray()
            for (osName in platformNames)
                dependsOn("$flavor-${osName}Jar")
        }
    }

    publishing.repositories {
        maven {
            if (project.hasProperty("snapshot")) {
                name = "snapshot"
                url = uri("https://central.sonatype.com/repository/maven-snapshots/")
                credentials {
                    username = findProperty("sonatypeUsername").toString()
                    password = findProperty("sonatypePassword").toString()
                }
            } else if (project.hasProperty("staging")) {
                name = "staging"
                url = uri("https://ossrh-staging-api.central.sonatype.com/service/local/staging/deploy/maven2/")
                credentials {
                    username = findProperty("sonatypeUsername").toString()
                    password = findProperty("sonatypePassword").toString()
                }
            } else {
                name = "local"
                url = uri("build/repo")
            }
        }
    }

    register("downloadMxnetNativeLib") {
        doLast {
            val url = "https://publish.djl.ai/mxnet-$mxnetVersion"
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
                val file = binaryRoot / value
                file.parentFile.mkdirs()
                "$url/$key".url gzipInto file
            }

            copy {
                from(binaryRoot / "mkl/linux/native/lib") {
                    exclude("**/libmxnet.so")
                }
                into(binaryRoot / "cu102mkl/linux/native/lib")
            }
            copy {
                from(binaryRoot / "mkl/linux/native/lib") {
                    exclude("**/libmxnet.so")
                }
                into(binaryRoot / "cu112mkl/linux/native/lib")
            }
        }
    }
}

// Post-publish task to make deployment visible in Central Publisher Portal.
// See https://central.sonatype.org/publish/publish-portal-ossrh-staging-api/#ensuring-deployment-visibility-in-the-central-publisher-portal
if (project.hasProperty("staging")) {
    val url = "https://ossrh-staging-api.central.sonatype.com/manual/upload/defaultRepository/${project.group}"
    val username = findProperty("sonatypeUsername").toString()
    val password = findProperty("sonatypePassword").toString()
    val token = Base64.getEncoder().encodeToString("${username}:${password}".toByteArray())

    tasks.register("postPublish") {
        doLast {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.requestMethod = "POST"
            conn.setRequestProperty("Authorization", "Bearer $token")
            val status = conn.responseCode
            if (status != HttpURLConnection.HTTP_OK) {
                project.logger.error("Failed to POST '${url}'. Received status code ${status}: ${conn.responseMessage}")
            }
        }
    }

    tasks.named("publish") {
        finalizedBy(tasks.named("postPublish"))
    }
}
