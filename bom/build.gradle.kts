import org.w3c.dom.Document
import org.w3c.dom.Element
import org.w3c.dom.Node
import org.w3c.dom.NodeList

plugins {
    `java-platform`
    `maven-publish`
    signing
}

group = "ai.djl"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
version = libs.versions.djl.get() + if (isRelease) "" else "-SNAPSHOT"

dependencies {
    constraints {
        api(projects.api)
        api(projects.basicdataset)
        api(projects.modelZoo)
        api(projects.djlZero)
        api("ai.djl.android:core:$version")
        api("ai.djl.android:onnxruntime:$version")
        api("ai.djl.android:pytorch-native:$version")
        api(projects.extension.audio)
        api(projects.extension.awsAi)
        api(projects.extension.fasttext)
        api(projects.extension.hadoop)
        api(projects.extension.opencv)
        api(projects.extension.sentencepiece)
        api(projects.extension.tablesaw)
        api(projects.extension.tokenizers)
        api(projects.engines.llama)
        api(projects.engines.ml.lightgbm)
        api(projects.engines.ml.xgboost)
        api("ai.djl.ml.xgboost:xgboost-gpu:$version")
        api(projects.engines.mxnet.mxnetEngine)
        api(projects.engines.mxnet.mxnetModelZoo)
        api(projects.engines.onnxruntime.onnxruntimeEngine)
        api(projects.engines.onnxruntime.onnxruntimeEngine)
        api("ai.djl.python:python:$version")
        api(projects.engines.pytorch.pytorchEngine)
        api(projects.engines.pytorch.pytorchJni)
        api(projects.engines.pytorch.pytorchModelZoo)
        api(projects.engines.tensorflow.tensorflowApi)
        api(projects.engines.tensorflow.tensorflowEngine)
        api(projects.engines.tensorflow.tensorflowModelZoo)
        api("ai.djl.spark:spark_2.12:$version")
        api("ai.djl.serving:prometheus:$version")
        api("ai.djl.serving:serving:$version")
        api("ai.djl.serving:wlm:$version")
        api("ai.djl.tensorrt:tensorrt:$version")
        api("ai.djl.timeseries:timeseries:$version")
        api("com.microsoft.onnxruntime:onnxruntime:${libs.versions.onnxruntime.get()}")
        api("com.microsoft.onnxruntime:onnxruntime_gpu:${libs.versions.onnxruntime.get()}")
    }
}

tasks.withType<GenerateModuleMetadata> { enabled = false }

publishing {
    publications {
        register<MavenPublication>("bom") {
            from(components["javaPlatform"])

            pom {
                name = "DJL Bill of Materials (BOM)"
                description = "Deep Java Library (DJL) Bill of Materials (BOM)"
                url = "http://www.djl.ai/bom"
                packaging = "pom"

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

                withXml {
                    operator fun Element.div(name: String): Element {
                        val nl = childNodes
                        for (i in 0..nl.length) {
                            val node = nl.item(i)
                            if (node.nodeName.endsWith(name)) return node as Element
                        }
                        error("element not found")
                    }

                    val pomNode = asElement()
                    val dependencies = pomNode / "dependencyManagement" / "dependencies"
                    dependencies.apply {
                        operator fun String.unaryPlus() {
                            val doc = ownerDocument
                            val dep = doc.createElement("dependency")
                            infix fun String.addWith(value: String) {
                                val node = doc.createElement(this)
                                node.appendChild(doc.createTextNode(value))
                                dep.appendChild(node)
                            }
                            val (g, a, v, c) = split(':')
                            "groupId" addWith g
                            "artifactId" addWith a
                            "version" addWith v
                            "classifier" addWith c
                            appendChild(dep)
                        }

                        val mxnet = libs.versions.mxnet.get()
                        +"ai.djl.mxnet:mxnet-native-mkl:$mxnet:osx-x86_64"
                        +"ai.djl.mxnet:mxnet-native-mkl:$mxnet:linux-x86_64"
                        +"ai.djl.mxnet:mxnet-native-mkl:$mxnet:win-x86_64"
                        +"ai.djl.mxnet:mxnet-native-cu112mkl:$mxnet:linux-x86_64"
                        var pytorch = libs.versions.pytorch.get()
                        +"ai.djl.pytorch:pytorch-native-cpu:$pytorch:osx-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cpu:$pytorch:osx-aarch64"
                        +"ai.djl.pytorch:pytorch-native-cpu:$pytorch:linux-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cpu:$pytorch:win-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cpu-precxx11:$pytorch:linux-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cpu-precxx11:$pytorch:linux-aarch64"
                        +"ai.djl.pytorch:pytorch-native-cu121:$pytorch:linux-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cu121:$pytorch:win-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cu121-precxx11:$pytorch:linux-x86_64"
                        pytorch = "1.13.1"
                        +"ai.djl.pytorch:pytorch-native-cu117:$pytorch:linux-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cu117:$pytorch:win-x86_64"
                        +"ai.djl.pytorch:pytorch-native-cu117-precxx11:$pytorch:linux-x86_64"
                        val tensorflow = libs.versions.tensorflow.get()
                        +"ai.djl.tensorflow:tensorflow-native-cpu:$tensorflow:osx-x86_64"
                        +"ai.djl.tensorflow:tensorflow-native-cpu:$tensorflow:linux-x86_64"
                        +"ai.djl.tensorflow:tensorflow-native-cpu:$tensorflow:win-x86_64"
                        +"ai.djl.tensorflow:tensorflow-native-cu113:$tensorflow:linux-x86_64"
                        +"ai.djl.tensorflow:tensorflow-native-cu113:$tensorflow:win-x86_64"
                    }
                }
            }
        }
    }

    repositories {
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
}

signing {
    isRequired = project.hasProperty("staging") || project.hasProperty("snapshot")
    val signingKey = findProperty("signingKey").toString()
    val signingPassword = findProperty("signingPassword").toString()
    useInMemoryPgpKeys(signingKey, signingPassword)
    sign(publishing.publications["bom"])
}