import java.util.regex.Pattern

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.mxnet"

dependencies {
    api(project(":api"))

    testImplementation(project(":basicdataset"))
    testImplementation(project(":model-zoo"))
    testImplementation(project(":testing"))

    testImplementation(libs.slf4j.simple)
}

sourceSets.main {
    java {
        srcDirs("src/main/java", "build/generated-src")
    }
}

tasks {
    processResources {
        val djlVersion = libs.versions.djl.get()
        val mxnetVersion = libs.versions.mxnet.get()
        inputs.properties(mapOf("djlVersion" to djlVersion, "mxnetVersion" to mxnetVersion))
        filesMatching("**/mxnet-engine.properties") {
            expand(mapOf("djlVersion" to djlVersion, "mxnetVersion" to mxnetVersion))
        }
    }

    checkstyleMain { exclude("ai/djl/mxnet/jna/*") }
    pmdMain { exclude("ai/djl/mxnet/jna/*") }

    val jnarator by registering {
        val jnaratorJar = project(":engines:mxnet:jnarator").tasks.jar
        dependsOn(jnaratorJar)
        inputs.file("${project.projectDir}/src/main/jna/mapping.properties")
            .withPathSensitivity(PathSensitivity.RELATIVE)
            .withPropertyName("jna/mapping.properties file")
        outputs.dir(buildDirectory / "generated-src")
        outputs.cacheIf { true }
        val jnaGenerator = jnaratorJar.get().outputs.files.singleFile
        val buildDir = buildDirectory
        val dir = projectDir
        val prov = providers

        doLast {
            prov.javaexec {
                workingDir = dir
                mainClass = "-jar"
                args(
                    jnaGenerator.absolutePath,
                    "-l",
                    "mxnet",
                    "-p",
                    "ai.djl.mxnet.jna",
                    "-o",
                    "$buildDir/generated-src",
                    "-m",
                    "${dir}/src/main/jna/mapping.properties",
                    "-f",
                    "src/main/include/mxnet/c_api.h",
                    "src/main/include/nnvm/c_api.h"
                )
            }.result.get()
        }
    }

    test {
        environment("PATH" to "src/test/bin:${environment["PATH"]}")
    }

    fun checkForUpdate(path: String, url: String) {
        val expected = url.url.text
        val actual = project.projectDir.resolve("src/main/include/$path").text
        if (actual != expected) {
            val fileName = path.replace("[/\\\\]", "_")
            val build = project.projectDir.resolve("build").apply { mkdirs() }
            build.resolve(fileName).text = expected
            logger.warn("""[\033[31mWARN\033[0m ] Header file has been changed in open source project: $path.""")
        }
    }

    register("checkHeaderFile") {
        outputs.files(buildDirectory / "mxnet_c_api.h", buildDirectory / "nnvm_c_api.h")
        doFirst {
            if (gradle.startParameter.isOffline) {
                logger.warn("""[\033[31mWARN\033[0m ] Ignore header validation in offline mode.""")
                return@doFirst
            }

            val mxnetUrl = "https://raw.githubusercontent.com/apache/incubator-mxnet/v1.7.x/"
            checkForUpdate("mxnet/c_api.h", "$mxnetUrl/include/mxnet/c_api.h")
            val content = "https://github.com/apache/incubator-mxnet/tree/v1.7.x/3rdparty".url.text

            val pattern = Pattern.compile("href=\"/apache/incubator-tvm/tree/([a-z0-9]+)\"")
            val m = pattern.matcher(content)
            if (!m.find())
                throw GradleException("Failed to retrieve submodule hash for tvm from github")
            val hash = m.group(1)

            val nnvmUrl = "https://raw.githubusercontent.com/apache/incubator-tvm/$hash"
            checkForUpdate("nnvm/c_api.h", "${nnvmUrl}/nnvm/include/nnvm/c_api.h")
        }
    }

    sourcesJar {
        duplicatesStrategy = DuplicatesStrategy.WARN
        dependsOn(jnarator)
    }
    compileJava { dependsOn(jnarator) }
    javadoc { dependsOn(jnarator) }
    verifyJava { dependsOn(jnarator) }

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
