@file:Suppress("UNCHECKED_CAST")

package ai.djl

import gradle.kotlin.dsl.accessors._47a1ba74e6c64ebd766d5f51da1685d2.*
import gradle.kotlin.dsl.accessors._47a1ba74e6c64ebd766d5f51da1685d2.javaToolchains
import org.gradle.kotlin.dsl.attributes
import org.gradle.kotlin.dsl.`java-library`
import org.gradle.kotlin.dsl.systemProperties

plugins {
    id("ai.djl.javaBase")
    `java-library`
    //    eclipse
    id("ai.djl.javaFormatter")
    id("ai.djl.check")
}

tasks {
    compileJava {
        javaCompiler = javaToolchains.compilerFor { languageVersion = JavaLanguageVersion.of(8) }
        options.apply {
            encoding = "UTF-8"
            compilerArgs = listOf("--release", "8", "-proc:none", "-Xlint:all,-options,-static", "-Werror")
        }
    }
    compileTestJava {
        javaCompiler = javaToolchains.compilerFor { languageVersion = JavaLanguageVersion.of(11) }
        options.apply {
            encoding = "UTF-8"
            compilerArgs = listOf("-proc:none", "-Xlint:all,-options,-static,-removal", "-Werror")
        }
    }
    javadoc {
        options {
            this as StandardJavadocDocletOptions // https://github.com/gradle/gradle/issues/7038
            addStringOption("Xdoclint:none", "-quiet")
        }
    }
    test {
        // tensorflow mobilenet and resnet require more cpu memory
        maxHeapSize = "4096m"

        useTestNG {
            //suiteXmlFiles = listOf(File(rootDir, "testng.xml")) //This is how to add custom testng.xml
        }

        testLogging {
            showStandardStreams = true
            events("passed", "skipped", "failed", "standardOut", "standardError")
        }

        doFirst {
            jvmArgs("--add-opens", "java.base/jdk.internal.loader=ALL-UNNAMED")
            systemProperties = System.getProperties().toMap() as Map<String, Any>
            systemProperties.remove("user.dir")
            // systemProperty "ai.djl.logging.level", "debug"
            systemProperties("org.slf4j.simpleLogger.defaultLogLevel" to "debug",
                             "org.slf4j.simpleLogger.log.org.mortbay.log" to "warn",
                             "org.slf4j.simpleLogger.log.org.testng" to "info",
                             "disableProgressBar" to "true",
                             "nightly" to System.getProperty("nightly", "false"))
            if (gradle.startParameter.isOffline)
                systemProperty("ai.djl.offline", "true")
            // This is used to avoid overriding on default engine for modules:
            // mxnet-engine, mxnet-model-zoo, api (MockEngine), basicdataset, fasttext, etc
            if (project.name != "integration" && project.name != "examples")
                systemProperties.remove("ai.djl.default_engine")
        }
    }
    jar {
        manifest {
            attributes("Automatic-Module-Name" to "ai.djl.${project.name.replace('-', '_')}",
                       "Specification-Version" to "${project.version}")
        }
    }
}

//eclipse {
//    jdt.file.withProperties {
//        setProperty("org.eclipse.jdt.core.circularClasspath", "warning")
//    }
//    classpath {
//        sourceSets.first { it.name == "test" }.java {
//            srcDirs("src/test/java")
//            exclude("**/package-info.java")
//        }
//    }
//}