rootProject.name = "djl"

val reserved = listOf("gradle", "buildSrc")
fun File.scan() {
    if (isDirectory && name !in reserved) {
        if (this != rootDir && resolve("build.gradle.kts").exists()) {
            val project = relativeTo(rootDir).toString()
            include(project.replace('/', ':'))
        }
        listFiles()!!.forEach { it.scan() }
    }
}
rootDir.scan()

dependencyResolutionManagement {
    repositories {
        mavenCentral()
        maven("https://oss.sonatype.org/content/repositories/snapshots/")
    }
}

enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")