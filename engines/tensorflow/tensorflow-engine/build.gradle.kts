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
    inputs.properties(mapOf("djlVersion" to libs.versions.djl.get(), "tensorflowVersion" to libs.versions.tensorflow.get()))
    filesMatching("**/tensorflow-engine.properties") {
        expand(mapOf("djlVersion" to libs.versions.djl.get(), "tensorflowVersion" to libs.versions.tensorflow.get()))
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
