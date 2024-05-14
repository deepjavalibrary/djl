import java.net.URL

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.ml.xgboost"
val isGpu = project.hasProperty("gpu")
val XGB_FLAVOR = if (isGpu) "-gpu" else ""

val exclusion by configurations.registering

dependencies {
    api(project(":api"))
    api("commons-logging:commons-logging:${libs.versions.commonsLogging.get()}")
    api("ml.dmlc:xgboost4j${XGB_FLAVOR}_2.12:${libs.versions.xgboost.get()}") {
        // get rid of the unused XGBoost Dependencies
        exclude("org.apache.hadoop", "hadoop-hdfs")
        exclude("org.apache.hadoop", "hadoop-common")
        exclude("junit", "junit")
        exclude("com.typesafe.akka", "akka-actor_2.12")
        exclude("com.typesafe.akka", "akka-testkit_2.12")
        exclude("com.esotericsoftware", "kryo")
        exclude("org.scalatest", "scalatest_2.12")
        exclude("org.scala-lang.modules", "scala-java8-compat_2.12")
        exclude("org.scala-lang", "scala-compiler")
        exclude("org.scala-lang", "scala-reflect")
        exclude("org.scala-lang", "scala-library")
    }

    exclusion(project(":api"))
    exclusion(libs.commons.logging)
    testImplementation(project(":testing"))

    testRuntimeOnly(libs.slf4j.simple)
    if (isGpu)
        testRuntimeOnly(libs.ai.rapids.cudf) { artifact { classifier = "cuda11" } }
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val jnilibDir = buildDirectory / "classes/java/main/lib/linux/aarch64/"
        outputs.dir(jnilibDir)
        doLast {
            val url = "https://publish.djl.ai/xgboost/${libs.versions.xgboost.get()}/jnilib/linux/aarch64/libxgboost4j.so"
            val file = jnilibDir / "libxgboost4j.so"
            if (!file.exists()) {
                project.logger.lifecycle("Downloading $url")
                url.url into file
            }
        }
    }

    jar {
        from((configurations.compileClasspath.get() - exclusion.get()).map {
            if (it.isDirectory())
                emptyList()
            else
                zipTree(it).matching {
                    include("lib/**",
                            "ml/dmlc/xgboost4j/java/DMatrix*",
                            "ml/dmlc/xgboost4j/java/NativeLibLoader*",
                            "ml/dmlc/xgboost4j/java/XGBoost*",
                            "ml/dmlc/xgboost4j/java/Column*",
                            "ml/dmlc/xgboost4j/java/util/*",
                            "ml/dmlc/xgboost4j/gpu/java/*",
                            "ml/dmlc/xgboost4j/LabeledPoint.*",
                            "xgboost4j-version.properties")
                }
        })
    }

    publishing {
        publications {
            named<MavenPublication>("maven") {
                artifactId = "${project.name}$XGB_FLAVOR"
                pom {
                    name = "DJL Engine Adapter for XGBoost"
                    description = "Deep Java Library (DJL) Engine Adapter for XGBoost"
                    url = "https://djl.ai/engines/ml/${project.name}"

                    withXml {
                        val pomNode = asNode()
                        //                        pomNode.dependencies."*".findAll() {
                        //                            it.artifactId.text().startsWith("xgboost")
                        //                        }.each() {
                        //                            it.parent().remove(it)
                        //                        }
                        //                        if (isGpu) {
                        //                            def dep = pomNode.dependencies[0].appendNode("dependency")
                        //                            dep.appendNode("groupId", "ai.rapids")
                        //                            dep.appendNode("artifactId", "cudf")
                        //                            dep.appendNode("version", "${rapis_version}")
                        //                            dep.appendNode("classifier", "cuda11")
                        //                            dep.appendNode("scope", "compile")
                        //                        }
                    }
                }
            }
        }
    }
}