package ai.djl

import org.gradle.kotlin.dsl.attributes
import org.gradle.kotlin.dsl.`java-library`
import org.gradle.kotlin.dsl.systemProperties

plugins {
    id("ai.djl.javaBase")
    `java-library`
    id("ai.djl.javaFormatter")
    id("ai.djl.check")
    id("ai.djl.stats")
}

tasks {
    compileJava {
        options.apply {
            release = 8
            encoding = "UTF-8"
            compilerArgs = listOf("-proc:none", "-Xlint:all,-options,-static", "-Werror")
        }
    }
    compileTestJava {
        options.apply {
            release = 11
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

        jvmArgs("--add-opens", "java.base/jdk.internal.loader=ALL-UNNAMED")
        for (prop in System.getProperties().iterator()) {
            val key = prop.key.toString()
            if (key.startsWith("ai.djl.")) {
                systemProperty(key, prop.value)
            }
        }
        systemProperties(
            "org.slf4j.simpleLogger.defaultLogLevel" to "debug",
            "org.slf4j.simpleLogger.log.org.mortbay.log" to "warn",
            "org.slf4j.simpleLogger.log.org.testng" to "info",
            "disableProgressBar" to "true",
            "nightly" to System.getProperty("nightly", "false"),
        )
        if (gradle.startParameter.isOffline)
            systemProperty("ai.djl.offline", "true")
        // This is used to avoid overriding on default engine for modules:
        // mxnet-engine, mxnet-model-zoo, api (MockEngine), basicdataset, fasttext, etc
        if (project.name != "integration" && project.name != "examples")
            systemProperties.remove("ai.djl.default_engine")
    }
    jar {
        manifest {
            attributes(
                "Automatic-Module-Name" to "ai.djl.${project.name.replace('-', '_')}",
                "Specification-Version" to "${project.version}"
            )
        }
    }
}
