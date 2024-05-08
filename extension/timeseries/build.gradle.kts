plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.timeseries"

dependencies {
    api(projects.api)
    api(projects.basicdataset)
    api(libs.tablesaw.core)
    api(libs.tablesaw.jsplot)

    testImplementation(libs.slf4j.simple)
    testImplementation(projects.testing)

    testRuntimeOnly(projects.engines.mxnet.mxnetModelZoo)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "TimeSeries for DJL"
                description = "TimeSeries for DJL"
                url = "http://www.djl.ai/extensions/${project.name}"
            }
        }
    }
}
