plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.opencv"

dependencies {
    api(projects.api)
    api(libs.openpnp.opencv)

    testImplementation(projects.testing)
    testRuntimeOnly(libs.apache.log4j.slf4j)

    testRuntimeOnly(projects.engines.pytorch.pytorchModelZoo)
    testRuntimeOnly(projects.engines.pytorch.pytorchJni)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "OpenCV toolkit for DJL"
                description = "OpenCV toolkit for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
