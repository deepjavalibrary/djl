package ai.djl

import buildDirectory
import com.github.spotbugs.snom.SpotBugsTask
import org.gradle.kotlin.dsl.*

plugins {
    pmd
    checkstyle
    jacoco
    id("com.github.spotbugs")
}

spotbugs {
    excludeFilter = file("${rootProject.projectDir}/tools/conf/findbugs-exclude.xml")
    ignoreFailures = false
}
tasks {
    named<SpotBugsTask>("spotbugsMain") {
        reports.create("html") {
            required = true
            outputLocation = file("$buildDirectory/reports/spotbugs.html")
        }
    }
    named<SpotBugsTask>("spotbugsTest") {
        enabled = true
        reports.create("html") {
            required = true
            outputLocation = file("$buildDirectory/reports/spotbugs.html")
        }
    }
}

pmd {
    isIgnoreFailures = false
    tasks["pmdTest"].enabled = false
    ruleSets = emptyList() // workaround pmd gradle plugin bug
    ruleSetFiles = files("${rootProject.projectDir}/tools/conf/pmd.xml")
}
tasks.withType<Pmd> {
    reports {
        xml.required = true
        html.required = true
    }
}

checkstyle {
    toolVersion = "10.14.2"
    isIgnoreFailures = false
    tasks["checkstyleTest"].enabled = true
    configProperties = mapOf(
        "checkstyle.suppressions.file" to file("${rootProject.projectDir}/tools/conf/suppressions.xml"),
        "checkstyle.licenseHeader.file" to file("${rootProject.projectDir}/tools/conf/licenseHeader.java")
    )
    configFile = file("${rootProject.projectDir}/tools/conf/checkstyle.xml")
}
tasks {
    named<Checkstyle>("checkstyleMain") {
        classpath += configurations["compileClasspath"]
    }
    withType<Checkstyle> {
        reports {
            xml.required = false
            html.required = true
        }
    }

    val jacocoTestReport = named<JacocoReport>("jacocoTestReport") {
        reports {
            xml.required = true
            csv.required = false
        }
    }
    named("test") { finalizedBy(jacocoTestReport) }
    named("build") { dependsOn(named("javadoc")) }
}
