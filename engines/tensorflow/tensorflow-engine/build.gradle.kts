plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorflow"

dependencies {
    api(project(":api"))
    api(project(":engines:tensorflow:tensorflow-api"))

    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

tasks.processResources {
    inputs.properties(mapOf("djlVersion" to libs.versions.djl, "tensorflowVersion" to libs.versions.tensorflow))
    filesMatching("**/mxnet-engine.properties") {
        expand(mapOf("djl_version" to libs.versions.djl, "tensorflow_version" to libs.versions.tensorflow))
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for TensorFlow"
                description = "Deep Java Library (DJL) Engine Adapter for TensorFlow"
                url = "http://www.djl.ai/engines/tensorflow/${project.name}"
            }
        }
    }
}
