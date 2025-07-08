plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.genai"

dependencies {
    api(project(":api"))

    testImplementation(project(":testing"))
    testRuntimeOnly(libs.slf4j.simple)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL generative ai extension"
                description = "DJL generative ai extension"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
