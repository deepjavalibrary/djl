package ai.djl

import div
import os
import text
import url

tasks {
    fun checkClang(clang: File) {
        val url = "https://djl-ai.s3.amazonaws.com/build-tools/osx/clang-format".url
        if (!clang.exists()) {
            // create the folder and download the executable
            clang.parentFile.mkdirs()
            // TODO original method was appending
            clang.writeBytes(url.openStream().use { it.readAllBytes() })
            clang.setExecutable(true)
        }
    }

    fun formatCpp(f: File, clang: File): String = when {
        !f.name.endsWith(".cc") && !f.name.endsWith(".cpp") && !f.name.endsWith(".h") -> ""
        else -> ProcessBuilder(
            clang.absolutePath,
            "-style={BasedOnStyle: Google, IndentWidth: 2, ColumnLimit: 120, AlignAfterOpenBracket: DontAlign, SpaceAfterCStyleCast: true}",
            f.absolutePath
        )
            .start().inputStream.use { it.readAllBytes().decodeToString() }
    }

    register("formatCpp") {
        val rootProject = project.rootProject
        val clang = rootProject.projectDir / ".clang/clang-format"
        val formatCpp = project.extensions.getByName<FormatterConfig>("formatCpp")
        val files = project.fileTree("src")
        val logger = project.logger

        doLast {
            if ("mac" !in os)
                return@doLast
            checkClang(clang)
            files.include("**/*.cc", "**/*.cpp", "**/*.h")
            if (formatCpp.exclusions.isPresent)
                for (exclusion in formatCpp.exclusions.get())
                    files.exclude(exclusion)

            for (f in files) {
                if (!f.isFile())
                    continue
                logger.info("formatting cpp file: $f")
                f.text = formatCpp(f, clang)
            }
        }
    }

    register("verifyCpp") {
        val rootProject = project.rootProject
        val clang = rootProject.projectDir / ".clang/clang-format"
        val files = project.fileTree("src")
        val logger = project.logger

        doLast {
            if ("mac" !in os)
                return@doLast
            checkClang(clang)
            files.include("**/*.cc", "**/*.cpp", "**/*.h")
            for (f in files) {
                if (!f.isFile())
                    continue
                logger.info("checking cpp file: $f")
                if (f.text != formatCpp(f, clang))
                    throw GradleException("File not formatted: " + f.absolutePath)
            }
        }
    }
}

interface FormatterConfig {
    val exclusions: ListProperty<String>
}
project.extensions.create<FormatterConfig>("formatCpp")

