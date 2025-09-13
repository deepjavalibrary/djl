package ai.djl

import com.google.googlejavaformat.java.Main
import org.gradle.kotlin.dsl.getByName
import org.gradle.kotlin.dsl.registering
import java.io.PrintWriter

tasks {
    register("formatJava") {
        val resultFilePath = "build/formatJava-result.txt"
        var files = project.sourceSets.flatMap { it.allSource }
        files = files.filter { it.name.endsWith(".java") && "generated-src" !in it.absolutePath }
        inputs.files(files)
        outputs.file(project.file(resultFilePath))
        doLast {
            val formatter = Main(PrintWriter(System.out, true), PrintWriter(System.err, true), System.`in`)
            for (f in files) {
                if (formatter.format("-a", "-i", f.absolutePath) != 0) {
                    throw GradleException("Format java failed: " + f.absolutePath)
                }
            }
            File(resultFilePath).writeText("Success")
        }
    }

    val verifyJava by registering {
        val resultFilePath = "build/verifyJava-result.txt"
        var files = project.sourceSets.flatMap { it.allSource }
        files = files.filter { it.name.endsWith(".java") && "generated-src" !in it.absolutePath }
        inputs.files(files)
        outputs.file(project.file(resultFilePath))
        doLast {
            val formatter = Main(PrintWriter(System.out, true), PrintWriter(System.err, true), System.`in`)
            for (f in files) {
                if (formatter.format("-a", "-n", "--set-exit-if-changed", f.absolutePath) != 0) {
                    throw GradleException(
                        "File not formatted: " + f.absolutePath
                                + System.lineSeparator()
                                + "In order to reformat your code, run './gradlew formatJava' (or './gradlew fJ' for short)"
                                + System.lineSeparator()
                                + "See https://github.com/deepjavalibrary/djl/blob/master/docs/development/development_guideline.md#coding-conventions for more details"
                    )
                }
            }
            File(resultFilePath).writeText("Success")
        }
    }

    named("check") { dependsOn(verifyJava) }
}

val Project.sourceSets: SourceSetContainer
    get() = extensions.getByName<SourceSetContainer>("sourceSets")
