import org.w3c.dom.Element

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorflow"

val exclusion by configurations.registering

@Suppress("UnstableApiUsage")
dependencies {
    api(libs.bytedeco.javacpp)
    api(libs.google.protobuf)
    api(libs.tensorflow.core)

    exclusion(libs.bytedeco.javacpp)
    exclusion(libs.google.protobuf)
}

tasks.jar {
    from((configurations.compileClasspath.get() - exclusion.get()).map {
        if (it.isDirectory()) it else zipTree(it)
    })
    exclude("module-info.class")
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL release for TensorFlow core api"
                description = "Deep Java Library (DJL) release for TensorFlow core api"
                url = "http://www.djl.ai/engines/tensorflow/${project.name}"

                withXml {
                    val pomNode = asElement()
                    val nl = pomNode.getElementsByTagName("dependency")
                    for (i in 0 until nl.length) {
                        val node = nl.item(i) as Element
                        val artifactId = node.getElementsByTagName("artifactId").item(0)
                        if (artifactId.textContent.startsWith("tensorflow-") || artifactId.textContent.startsWith("ndarray")) {
                            val dependencies = node.parentNode
                            dependencies.removeChild(node)
                        }
                    }
                }
            }
        }
    }
}
