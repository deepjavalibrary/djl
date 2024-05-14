plugins {
    ai.djl.javaProject
}

dependencies {
    api(project(":api"))
    // TODO Could not resolve org.testng:testng:7.9.0.
//    api(libs.testng) { exclude("junit", "junit") }
}

tasks.compileJava {
    javaCompiler = javaToolchains.compilerFor { languageVersion = JavaLanguageVersion.of(11) }
    options.apply { compilerArgs = compilerArgs - listOf("--release", "8") }
}
