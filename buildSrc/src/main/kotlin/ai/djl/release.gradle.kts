package ai.djl

import org.gradle.accessors.dm.LibrariesForLibs
import org.gradle.kotlin.dsl.the

import text

val libs = the<LibrariesForLibs>()

tasks {
    register("increaseBuildVersion") {
        doLast {
            if (!project.hasProperty("targetVersion")) {
                throw GradleException("targetVersion property is required.")
            }
            val djlVersion = libs.versions.djl.get()
            val targetVersion = project.property("targetVersion")

            var file = file("examples/pom.xml")
            file.text = file.text.replace(
                "<djl.version>${djlVersion}-SNAPSHOT</djl.version>",
                "<djl.version>${targetVersion}-SNAPSHOT</djl.version>"
            ).replace(
                "<version>${djlVersion}-SNAPSHOT</version>",
                "<version>${targetVersion}-SNAPSHOT</version>"
            )

            file = file("gradle/libs.versions.toml")
            file.text = file.text.replace("djl = \"${djlVersion}\"", "djl = \"${targetVersion}\"")

            file = file("api/README.md")
            file.text = file.text.replace(
                "<version>${djlVersion}-SNAPSHOT</version>",
                "<version>${targetVersion}-SNAPSHOT</version>"
            )
        }
    }

    register("increaseFinalVersion") {
        doLast {
            if (!project.hasProperty("previousVersion")) {
                throw GradleException("previousVersion property is required.")
            }
            val previousVersion = project.property("previousVersion")

            val collection = fileTree(".").filter {
                it.name.endsWith(".md") || it.name.endsWith("overview.html")
            }
            val djlVersion = libs.versions.djl.get()

            for (file in collection) {
                file.text = file.text.replace("/${previousVersion}/", "/${djlVersion}/")
                    .replace(">${previousVersion}<", ">${djlVersion}<")
                    .replace("\"${previousVersion}\"", "\"${djlVersion}\"")
                    .replace("-${previousVersion}", "-${djlVersion}")
                    .replace("_${previousVersion}", "_${djlVersion}")
                    .replace(":${previousVersion}([\"\\\\\\n])", ":${djlVersion}\$1")
            }
        }
    }
}
