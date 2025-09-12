import org.w3c.dom.Element

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.ml.xgboost"
val isGpu = project.hasProperty("gpu")
val xgbFlavor = if (isGpu) "-gpu" else ""

val exclusion by configurations.registering

dependencies {
    api(project(":api"))
    api(libs.commons.logging)
    api("ml.dmlc:xgboost4j${xgbFlavor}_2.12:${libs.versions.xgboost.get()}") {
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

    jar {
        val retained = configurations.compileClasspath.get() - exclusion.get()
        for (file in retained)
            if (file.isFile)
                from(zipTree(file).matching {
                    include(
                        "lib/**",
                        "ml/dmlc/xgboost4j/java/DMatrix*",
                        "ml/dmlc/xgboost4j/java/NativeLibLoader*",
                        "ml/dmlc/xgboost4j/java/XGBoost*",
                        "ml/dmlc/xgboost4j/java/Column*",
                        "ml/dmlc/xgboost4j/java/util/*",
                        "ml/dmlc/xgboost4j/gpu/java/*",
                        "ml/dmlc/xgboost4j/LabeledPoint.*",
                        "xgboost4j-version.properties"
                    )
                })
    }

    publishing {
        val rapisVersion = libs.versions.rapis.get()
        val projectName = project.name
        val xgbFlavor = xgbFlavor
        val isGpu = isGpu

        publications {
            named<MavenPublication>("maven") {
                artifactId = "${projectName}$xgbFlavor"
                pom {
                    name = "DJL Engine Adapter for XGBoost"
                    description = "Deep Java Library (DJL) Engine Adapter for XGBoost"
                    url = "https://djl.ai/engines/ml/${projectName}"

                    withXml {
                        val pomNode = asElement()
                        val nl = pomNode.getElementsByTagName("dependency")
                        for (i in 0 until nl.length) {
                            val node = nl.item(i) as Element
                            val artifactId = node.getElementsByTagName("artifactId").item(0)
                            if (artifactId.textContent.startsWith("xgboost")) {
                                val dependencies = node.parentNode
                                dependencies.removeChild(node)
                                if (isGpu) {
                                    val doc = pomNode.ownerDocument
                                    val dep = doc.createElement("dependency")
                                    dependencies.appendChild(dep)
                                    infix fun String.addWith(value: String) {
                                        val n = doc.createElement(this)
                                        n.appendChild(doc.createTextNode(value))
                                        dep.appendChild(n)
                                    }
                                    "groupId" addWith "ai.rapids"
                                    "artifactId" addWith "cudf"
                                    "version" addWith rapisVersion
                                    "classifier" addWith "cuda11"
                                    "scope" addWith "compile"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
