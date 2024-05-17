plugins {
    ai.djl.javaProject
}

dependencies {
    api(project(":api"))
    api(libs.testng) { exclude("junit", "junit") }
}

tasks.compileJava {
//    javaCompiler = javaToolchains.compilerFor { languageVersion = JavaLanguageVersion.of(11) }
    options.apply {
        release = 11
//        compilerArgs = compilerArgs - listOf("--release", "8")
    }
}
