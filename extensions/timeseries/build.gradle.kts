plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.timeseries"

dependencies {
    api(project(":api"))
    api(project(":basicdataset"))
    api(libs.tablesaw.core)
    api(libs.tablesaw.jsplot)

    testImplementation(libs.slf4j.simple)
    testImplementation(project(":testing"))

    testRuntimeOnly(project(":engines:mxnet:mxnet-model-zoo"))
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
