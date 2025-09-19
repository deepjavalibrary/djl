plugins {
    ai.djl.javaProject
    antlr
}

dependencies {
    antlr(libs.antlr)

    api(libs.commons.cli)
    api(libs.antlrRuntime)
    api(libs.apache.log4j.slf4j)

    testImplementation(libs.testng)
}

tasks {
    checkstyleMain { exclude("ai/djl/mxnet/jnarator/parser/*") }
    pmdMain { exclude("ai/djl/mxnet/jnarator/parser/*") }

    compileJava {
        options.apply {
            release = 8
            encoding = "UTF-8"
            compilerArgs = listOf("-proc:none", "-Xlint:all,-options,-static")
        }
    }

    jar {
        dependsOn(generateGrammarSource)
        manifest {
            attributes(
                "Main-Class" to "ai.djl.mxnet.jnarator.Main",
                "Multi-Release" to true
            )
        }
        includeEmptyDirs = false
        duplicatesStrategy = DuplicatesStrategy.INCLUDE
        from(configurations.runtimeClasspath.get().map { if (it.isDirectory()) it else zipTree(it) })
    }
}
