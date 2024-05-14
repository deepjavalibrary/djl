plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.ml.lightgbm"

dependencies {
    api(project(":api"))
    api("com.microsoft.ml.lightgbm:lightgbmlib:${libs.versions.lightgbm.get()}")

    testImplementation(project(":testing"))

    testRuntimeOnly(libs.slf4j.simple)
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for LightGBM"
                description = "Deep Java Library (DJL) Engine Adapter for LightGBM"
                url = "https://djl.ai/engines/ml/${project.name}"
            }
        }
    }
}
