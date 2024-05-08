plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorflow"

val exclusion by configurations.registering

dependencies {
    api(libs.bytedeco.javacpp)
    api(libs.google.protobuf)
    api(libs.tensorflow)

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
                    val pomNode = asNode()
//                    pomNode.dependencies."*".findAll() {
//                        it.artifactId.text().startsWith("tensorflow-") || it.artifactId.text().startsWith("ndarray")
//                    }.each() {
//                        it.parent().remove(it)
//                    }
                }
            }
        }
    }
}
