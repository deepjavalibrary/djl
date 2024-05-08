import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(projects.api)
    api(libs.apache.commons.csv)

    // Add following dependency to your project for COCO dataset
    // runtimeOnly(libs.twelvemonkeys.imageio)
    testImplementation(libs.slf4j.simple)

    testImplementation(projects.testing)
    testImplementation(projects.engines.pytorch.pytorchEngine)
}

tasks.register<Exec>("syncS3") {
    commandLine("sh", "-c", "find . -name .DS_Store | xargs rm && aws s3 sync src/test/resources/mlrepo s3://djl-ai/mlrepo --acl public-read")
    standardOutput = ByteArrayOutputStream()
    ext["output"] = { standardOutput.toString() }
}
