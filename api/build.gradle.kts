plugins {
    ai.djl.javaProject
    ai.djl.publish
}

dependencies {
    api(libs.google.gson)
    api(libs.jna)
    api(libs.apache.commons.compress) {
        exclude("org.apache.commons", "commons-lang3")
    }
    api(libs.slf4j.api)

    testImplementation(project(":testing"))
    testImplementation(libs.testng)
    testImplementation(libs.slf4j.simple)
    testRuntimeOnly(project(":basicdataset"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val version = project.version
        inputs.properties(mapOf("version" to version))
        filesMatching("**/api.properties") {
            expand(mapOf("version" to version))
        }
    }

    javadoc {
        title = "Deep Java Library ${project.version} API specification"
        exclude("ai/djl/util/**", "ai/djl/ndarray/internal/**")
    }

    jar {
        manifest {
            attributes(
                "Notice" to "DJL will collect telemetry to help us better understand our users'" +
                        " needs, diagnose issues, and deliver additional features. If you would" +
                        " like to learn more or opt-out please go to: " +
                        "https://docs.djl.ai/master/docs/telemetry.html for more information."
            )
        }
    }
}
