plugins {
    `kotlin-dsl`
}

repositories {
    mavenCentral()
    gradlePluginPortal()
}

dependencies {
    implementation("com.google.googlejavaformat:google-java-format:1.15.0")
    implementation("com.github.spotbugs.snom:spotbugs-gradle-plugin:5.1.3")
    implementation(files(libs.javaClass.superclass.protectionDomain.codeSource.location))
}