plugins {
    ai.djl.javaProject
}

dependencies {
    api(project(":api"))
    api(libs.testng) { exclude("junit", "junit") }
}

tasks.compileJava {
    options.apply {
        release = 11
    }
}
