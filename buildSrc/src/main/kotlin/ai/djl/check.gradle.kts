package ai.djl

import com.github.spotbugs.snom.SpotBugsExtension
import com.github.spotbugs.snom.SpotBugsTask
import org.gradle.kotlin.dsl.*

plugins {
    pmd
    checkstyle
    jacoco
}

if (JavaVersion.current() < JavaVersion.VERSION_19) {
    apply(plugin = "com.github.spotbugs")
    spotbugs {
        excludeFilter = file("${rootProject.projectDir}/tools/conf/findbugs-exclude.xml")
        ignoreFailures = false
        //        spotbugsTest.enabled = true
    }
    tasks {
        named<SpotBugsTask>("spotbugsMain") {
            reports {
                // project ':api': Tasks collecting failure
                // Caused by: org.gradle.api.UnknownDomainObjectException: SpotBugsReport with name 'xml' not found
                //named("xml") { enabled = false }
                //named("html") { enabled = true }
            }
        }
        named<SpotBugsTask>("spotbugsTest") {
            enabled = true
            reports {
                named("xml") { enabled = false }
                named("html") { enabled = true }
            }
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
        "checkstyle.licenseHeader.file" to file("${rootProject.projectDir}/tools/conf/licenseHeader.java"))
    configFile = file("${rootProject.projectDir}/tools/conf/checkstyle.xml")
}
tasks {
    named<Checkstyle>("checkstyleMain") {
        classpath += configurations["compileClasspath"]
    }
    tasks.withType<Checkstyle> {
        reports {
            xml.required = false
            html.required = true
        }
    }
}

tasks {
    val jacocoTestReport = named<JacocoReport>("jacocoTestReport") {
        reports {
            xml.required = true
            csv.required = false
        }
    }
    named("test") { finalizedBy(jacocoTestReport) }
    named("build") { dependsOn(named("javadoc")) }

}

fun spotbugs(configure: SpotBugsExtension.() -> Unit) {
    val ext = extensions["spotbugs"] as SpotBugsExtension
    ext.configure()
}