import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.pytorch"

dependencies {
    api(project(":engines:pytorch:pytorch-engine"))

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.slf4j.simple)
    testImplementation(project(":testing"))
}

tasks.register<Exec>("syncS3") {
    commandLine("sh", "-c", "find . -name .DS_Store | xargs rm && aws s3 sync src/test/resources/mlrepo s3://djl-ai/mlrepo --acl public-read")

    standardOutput = ByteArrayOutputStream()
    ext["output"] = { standardOutput.toString() }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL model zoo for PyTorch"
                description = "Deep Java Library (DJL) model zoo for PyTorch"
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}
