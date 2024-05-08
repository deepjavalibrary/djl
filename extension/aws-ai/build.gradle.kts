plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.aws"

dependencies {
    api(platform(libs.awssdk.bom))
    api(libs.awssdk.s3)
    api(projects.api)

    testImplementation(projects.engines.pytorch.pytorchModelZoo)

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.apache.log4j.slf4j)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "AWS AI toolkit for DJL"
                description = "AWS AI toolkit for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
