import java.io.ByteArrayOutputStream

plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.mxnet"

dependencies {
    api(project(":api"))
    api(project(":engines:mxnet:mxnet-engine"))

    testImplementation(project(":basicdataset"))
    testImplementation(project(":model-zoo"))
    testImplementation(project(":testing"))

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
                name = "DJL model zoo for Apache MXNet"
                description = "Deep Java Library (DJL) model zoo for Apache MXNet"
                url = "http://www.djl.ai/engines/mxnet/${project.name}"
            }
        }
    }
}
