package ai.djl

import com.google.googlejavaformat.java.Main
import org.gradle.kotlin.dsl.getByName
import org.gradle.kotlin.dsl.registering
import java.io.PrintWriter

tasks {
    register("formatJava") {
        val sourceSets = project.sourceSets
        doLast {
            val formatter = Main(PrintWriter(System.out, true), PrintWriter(System.err, true), System.`in`)
            for (item in sourceSets)
                for (file in item.allSource) {
                    if (!file.name.endsWith(".java") || "generated-src" in file.absolutePath)
                        continue
                    if (formatter.format("-a", "-i", file.absolutePath) != 0)
                        throw GradleException("Format java failed: " + file.absolutePath)
                }
        }
    }

    val verifyJava by registering {
        val resultFilePath = "build/verifyJava-result.txt"
        inputs.files(project.sourceSets.flatMap { it.allSource })
        inputs.files(project.fileTree("generated-src"))
        outputs.file(project.file(resultFilePath))

        val proj = project
        doLast {
            val formatter = Main(PrintWriter(System.out, true), PrintWriter(System.err, true), System.`in`)
            for (item in proj.sourceSets)
                for (file in item.allSource) {
                    if (!file.name.endsWith(".java") || "generated-src" in file.absolutePath)
                        continue
                    if (formatter.format("-a", "-n", "--set-exit-if-changed", file.absolutePath) != 0)
                        throw GradleException(
                            "File not formatted: " + file.absolutePath
                                    + System.lineSeparator()
                                    + "In order to reformat your code, run './gradlew formatJava' (or './gradlew fJ' for short)"
                                    + System.lineSeparator()
                                    + "See https://github.com/deepjavalibrary/djl/blob/master/docs/development/development_guideline.md#coding-conventions for more details"
                        )
                }
            proj.file(resultFilePath).writeText("Success")
        }
    }

    named("check") { dependsOn(verifyJava) }
}

val Project.sourceSets: SourceSetContainer
    get() = extensions.getByName<SourceSetContainer>("sourceSets")
