import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorflow"

dependencies {
    api(projects.engines.tensorflow.tensorflowEngine)

    testImplementation(projects.testing)
    testImplementation(libs.slf4j.simple)
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
                name = "DJL model zoo for TensorFlow"
                description = "DJL model zoo for TensorFlow"
                url = "http://www.djl.ai/engines/tensorflow/${project.name}"
            }
        }
    }
}
