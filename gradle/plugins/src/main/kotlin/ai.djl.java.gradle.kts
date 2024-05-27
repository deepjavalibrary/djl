plugins {
    java
    idea
}

group = "ai.djl"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
//version = "${djl_version}" + (isRelease ? "" : "-SNAPSHOT")

repositories {
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}

idea {
    module {
        outputDir = file("build/classes/java/main")
        testOutputDir = file("build/classes/java/test")
        // inheritOutputDirs = true
    }
}