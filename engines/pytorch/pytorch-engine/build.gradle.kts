plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.pytorch"

dependencies {
    api(project(":api"))

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
    testImplementation(libs.slf4j.simple)
    testImplementation(project(":testing"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-model-zoo"))
    testRuntimeOnly(project(":engines:pytorch:pytorch-jni"))
}

tasks {
    compileJava { dependsOn(processResources) }

    processResources {
        outputs.file(buildDirectory / "classes/java/main/pytorch-engine.properties")
        doFirst {
            val classesDir = buildDirectory / "classes/java/main/"
            classesDir.mkdirs()
            val propFile = classesDir / "pytorch-engine.properties"
            propFile.text = "djl_version=${libs.versions.djl.get()}\npytorch_version=${libs.versions.pytorch.get()}"
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