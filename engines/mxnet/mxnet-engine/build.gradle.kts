import java.net.URL
import java.util.regex.Pattern

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.mxnet"

dependencies {
    api(projects.api)

    testImplementation(projects.basicdataset)
    testImplementation(projects.modelZoo)
    testImplementation(projects.testing)

    testImplementation(libs.slf4j.simple)
}

sourceSets.main {
    java {
        srcDirs("src/main/java", "build/generated-src")
    }
}

tasks {
    processResources {
        doFirst {
            val classesDir = layout.buildDirectory / "classes/java/main/"
            classesDir.mkdirs()
            val file = classesDir / "mxnet-engine.properties"
            file.text = "djl_version=${libs.versions.djl.get()}\nmxnet_version=" + libs.versions.mxnet.get()
        }
    }

    checkstyleMain { source("src/main/java") }
    pmdMain { source("src/main/java") }

    val jnarator by registering {
        val jnaratorJar = project(":engines:mxnet:jnarator").tasks.jar
        dependsOn(jnaratorJar)
        outputs.dir(layout.buildDirectory / "generated-src")
        doLast {
            val jnaGenerator = jnaratorJar.get().outputs.files.singleFile
            javaexec {
                mainClass = "-jar"
                args(jnaGenerator.absolutePath,
                     "-l",
                     "mxnet",
                     "-p",
                     "ai.djl.mxnet.jna",
                     "-o",
                     "${layout.buildDirectory}/generated-src",
                     "-m",
                     "${project.projectDir}/src/main/jna/mapping.properties",
                     "-f",
                     "src/main/include/mxnet/c_api.h",
                     "src/main/include/nnvm/c_api.h")
            }
        }
    }

    test {
        environment("PATH" to "src/test/bin:${environment["PATH"]}")
    }

    fun checkForUpdate(path: String, url: String) {
        val expected = URL(url).text
        val actual = project.projectDir.resolve("src/main/include/$path").text
        if (actual != expected) {
            val fileName = path.replace("[/\\\\]", "_")
            val build = project.projectDir.resolve("build").apply { mkdirs() }
            build.resolve(fileName).text = expected
            logger.warn("""[\033[31mWARN\033[0m ] Header file has been changed in open source project: $path.""")
        }
    }

    register("checkHeaderFile") {
        outputs.files(layout.buildDirectory / "mxnet_c_api.h", layout.buildDirectory / "nnvm_c_api.h")
        doFirst {
            if (gradle.startParameter.isOffline) {
                logger.warn("""[\033[31mWARN\033[0m ] Ignore header validation in offline mode.""")
                return@doFirst
            }

            val mxnetUrl = "https://raw.githubusercontent.com/apache/incubator-mxnet/v1.7.x/"
            checkForUpdate("mxnet/c_api.h", "$mxnetUrl/include/mxnet/c_api.h")
            val content = URL("https://github.com/apache/incubator-mxnet/tree/v1.7.x/3rdparty").text

            val pattern = Pattern.compile("href=\"/apache/incubator-tvm/tree/([a-z0-9]+)\"")
            val m = pattern.matcher(content)
            if (!m.find())
                throw GradleException("Failed to retrieve submodule hash for tvm from github")
            val hash = m.group(1)

            val nnvmUrl = "https://raw.githubusercontent.com/apache/incubator-tvm/$hash"
            checkForUpdate("nnvm/c_api.h", "${nnvmUrl}/nnvm/include/nnvm/c_api.h")
        }
    }

    sourcesJar { dependsOn(jnarator) }
    compileJava { dependsOn(jnarator) }
    javadoc { dependsOn(jnarator) }

    publishing {
        publications {
            named<MavenPublication>("maven") {
                pom {
                    name = "DJL Engine Adapter for Apache MXNet"
                    description = "Deep Java Library (DJL) Engine Adapter for Apache MXNet"
                    url = "http://www.djl.ai/engines/mxnet/${project.name}"
                }
            }
        }
    }
}