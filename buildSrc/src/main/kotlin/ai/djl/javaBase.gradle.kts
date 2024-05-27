package ai.djl

import org.gradle.accessors.dm.LibrariesForLibs
import org.gradle.kotlin.dsl.java
import org.gradle.kotlin.dsl.maven
import org.gradle.kotlin.dsl.repositories
import org.gradle.kotlin.dsl.the

plugins {
    java
}

group = "ai.djl"
val isRelease = project.hasProperty("release") || project.hasProperty("staging")
val libs = the<LibrariesForLibs>()
version = libs.versions.djl.get() + if (isRelease) "" else "-SNAPSHOT"

repositories {
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots/")
}
