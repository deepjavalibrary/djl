plugins {
    ai.djl.javaProject
}

dependencies {
    api(project(":api"))
    api(libs.testng)
}

tasks.compileJava {
    options.apply {
        release = 11
    }
}
