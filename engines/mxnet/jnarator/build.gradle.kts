plugins {
    ai.djl.javaProject
    antlr
}

dependencies {
    antlr(libs.antlr)

    api(libs.commons.cli)
    api(libs.antlrRuntime)
    api(libs.apache.log4j.slf4j)

    testImplementation(libs.testng) {
        exclude("junit", "junit")
    }
}

tasks {
    checkstyleMain { source("src/main/java") }
    pmdMain { source("src/main/java") }

    compileJava {
        // this crashes the build, because probably the returned object is a (copy of a) `List`
//        options.compilerArgs.clear()
        options.compilerArgs = listOf("--release", "11", "-proc:none", "-Xlint:all,-options,-static")
    }

    jar {
        manifest {
            attributes("Main-Class" to "ai.djl.mxnet.jnarator.Main",
                       "Multi-Release" to true)
        }
        includeEmptyDirs = false
        duplicatesStrategy = DuplicatesStrategy.INCLUDE
        from(configurations.runtimeClasspath.get().map { if (it.isDirectory()) it else zipTree(it) })
    }
}