plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tablesaw"

dependencies {
    api(projects.api)
    api(projects.basicdataset)
    api(libs.tablesaw.core)

    testImplementation(libs.slf4j.simple)
    testImplementation(projects.testing)

    testRuntimeOnly(libs.tablesaw.excel)
    testRuntimeOnly(libs.tablesaw.html)
    testRuntimeOnly(libs.tablesaw.json)
    testRuntimeOnly(projects.engines.pytorch.pytorchEngine)
    testRuntimeOnly(projects.engines.pytorch.pytorchJni)
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
