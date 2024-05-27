plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tablesaw"

dependencies {
    api(project(":api"))
    api(project(":basicdataset"))
    api(libs.tablesaw.core)

    testImplementation(libs.slf4j.simple)
    testImplementation(project(":testing"))

    testRuntimeOnly(libs.tablesaw.excel)
    testRuntimeOnly(libs.tablesaw.html)
    testRuntimeOnly(libs.tablesaw.json)
    testRuntimeOnly(project(":engines:pytorch:pytorch-engine"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "Tablesaw toolkit for DJL"
                description = "Tablesaw toolkit for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
