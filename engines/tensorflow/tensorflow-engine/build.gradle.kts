plugins {
    ai.djl.javaProject
    ai.djl.publish
}

group = "ai.djl.tensorflow"

dependencies {
    api(project(":api"))
    api(project(":engines:tensorflow:tensorflow-api"))

    testImplementation(project(":testing"))
    testImplementation(libs.slf4j.simple)
}

tasks.processResources {
    doFirst {
        val classesDir = buildDirectory / "classes/java/main/"
        classesDir.mkdirs()
        val file = classesDir / "tensorflow-engine.properties"
        file.text = "djl_version=${libs.versions.djl.get()}\ntensorflow_version=${libs.versions.tensorflow.get()}"
    }
}

publishing {
    publications {
        named<MavenPublication>("maven") {
            pom {
                name = "DJL Engine Adapter for TensorFlow"
                description = "Deep Java Library (DJL) Engine Adapter for TensorFlow"
                url = "http://www.djl.ai/engines/tensorflow/${project.name}"
            }
        }
    }
}
