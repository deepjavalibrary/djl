plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.google"

dependencies {
    api(platform(libs.google.cloud.bom))
    api(libs.google.cloud.storage)
    api(project(":api"))

    testImplementation(project(":engines:pytorch:pytorch-model-zoo"))

    testImplementation(libs.testng)
    testImplementation(libs.mockito)
    testImplementation(libs.apache.log4j.slf4j)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "Google Cloud toolkit for DJL"
                description = "Google Cloud toolkit for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
