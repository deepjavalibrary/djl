import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(projects.api)

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.slf4j.simple)
}

tasks.register<Exec>("syncS3") {
    commandLine("sh", "-c", "find . -name .DS_Store | xargs rm && aws s3 sync src/test/resources/mlrepo s3://djl-ai/mlrepo --acl public-read")

    standardOutput = ByteArrayOutputStream()
    ext["output"] = { standardOutput.toString() }
}
