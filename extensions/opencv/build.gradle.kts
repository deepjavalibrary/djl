plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.opencv"

dependencies {
    api(project(":api"))
    api(libs.openpnp.opencv)

    testImplementation(project(":testing"))
    testRuntimeOnly(libs.apache.log4j.slf4j)

    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
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
