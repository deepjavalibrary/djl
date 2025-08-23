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
    val djlVersion = libs.versions.djl.get()
    val tensorflowVersion = libs.versions.tensorflow.get()
    inputs.properties(mapOf("djlVersion" to djlVersion, "tensorflowVersion" to tensorflowVersion))
    filesMatching("**/tensorflow-engine.properties") {
        expand(mapOf("djlVersion" to djlVersion, "tensorflowVersion" to tensorflowVersion))
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
