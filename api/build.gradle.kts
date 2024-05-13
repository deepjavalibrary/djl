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

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.slf4j.simple)
    testRuntimeOnly(projects.engines.pytorch.pytorchModelZoo)
    testRuntimeOnly(projects.engines.pytorch.pytorchJni)
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.file(layout.buildDirectory / "classes/java/main/ai/djl/engine/api.properties")
        doFirst {
            val classesDir = file("${layout.buildDirectory}/classes/java/main/ai/djl/engine/")
            classesDir.mkdirs()
            val propFile = File(classesDir, "api.properties")
            propFile.text = "djl_version=${project.version}"
        }
    }

    javadoc {
        title = "Deep Java Library ${project.version} API specification"
        exclude("ai/djl/util/**", "ai/djl/ndarray/internal/**")
    }

    jar {
        manifest {
            attributes("Notice" to """DJL will collect telemetry to help us better understand our users’ needs, 
                |diagnose issues, and deliver additional features. If you would like to learn more or opt-out please go 
                |to: https://docs.djl.ai/docs/telemetry.html for more information.""".trimMargin())
        }
    }
}