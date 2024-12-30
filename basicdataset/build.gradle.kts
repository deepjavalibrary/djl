import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(project(":api"))
    api(libs.apache.commons.csv)

    // Add following dependency to your project for COCO dataset
    // runtimeOnly(libs.twelvemonkeys.imageio)
    testImplementation(libs.slf4j.simple)

    testImplementation(project(":testing"))
    testImplementation(project(":engines:pytorch:pytorch-engine"))
}

tasks.register<Exec>("syncS3") {
    workingDir = project.projectDir
    commandLine(
        "sh",
        "-c",
        "find . -name .DS_Store | xargs rm && aws s3 sync src/test/resources/mlrepo s3://djl-ai/mlrepo --acl public-read"
    )
    standardOutput = ByteArrayOutputStream()
    ext["output"] = { standardOutput.toString() }
}
