plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.pytorch"

dependencies {
    api(project(":api"))

    testImplementation(libs.testng)
    testImplementation(libs.slf4j.simple)
    testImplementation(project(":testing"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        val djlVersion = libs.versions.djl.get()
        val pytorchVersion = libs.versions.pytorch.get()
        inputs.properties(mapOf("djlVersion" to djlVersion, "pytorchVersion" to pytorchVersion))
        filesMatching("**/pytorch-engine.properties") {
            expand(mapOf("djlVersion" to djlVersion, "pytorchVersion" to pytorchVersion))
        }
    }

    test {
        environment("PATH" to "src/test/bin:${environment["PATH"]}")
    }

    clean { dependsOn(":engines:pytorch:pytorch-jni:clean") }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for PyTorch"
                description = "Deep Java Library (DJL) Engine Adapter for PyTorch"
                url = "http://www.djl.ai/engines/pytorch/${project.name}"
            }
        }
    }
}
