plugins {
    id ("com.github.spotbugs")/* version "5.1.3"*/ apply false
}

defaultTasks("build")

//apply from: file("${rootProject.projectDir}/tools/gradle/release.gradle")
//apply from: file("${rootProject.projectDir}/tools/gradle/stats.gradle")
