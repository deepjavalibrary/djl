package ai.djl

import java.util.*
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

plugins {
    `java-library`
}

val testsResults = TreeMap<Duration, String>(Comparator.reverseOrder())

tasks.test {

    addTestListener(object : TestListener {
        override fun beforeSuite(suite: TestDescriptor) {}
        override fun afterSuite(suite: TestDescriptor, result: TestResult) {}
        override fun beforeTest(testDescriptor: TestDescriptor) {}
        override fun afterTest(testDescriptor: TestDescriptor, result: TestResult) {
            val duration = (result.endTime - result.startTime).seconds
            testsResults[duration] = testDescriptor.className + '.' + testDescriptor.name
        }
    })

    doLast {
        if ("build" in gradle.startParameter.taskNames && testsResults.isNotEmpty()) {
            println("========== Test duration ==========")
            for((value, key) in testsResults.entries.take(5))
                println("\t$value:\t${key}s")
        }
    }
}