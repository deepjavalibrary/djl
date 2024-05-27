@file:OptIn(ExperimentalTime::class)

package ai.djl

import org.gradle.kotlin.dsl.support.serviceOf
import java.util.*
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

plugins {
    `java-library`
}

val timeSource = TimeSource.Monotonic
val testsResults = TreeMap<Duration, String>(Comparator.reverseOrder())

tasks.test {
    doFirst { startTime = timeSource.markNow() }
    doLast {
        if (state.didWork)
            testsResults[startTime - timeSource.markNow()] = project.name
    }
}

class DoSomething : FlowAction<FlowParameters.None> {
    override fun execute(parameters: FlowParameters.None) {
        if ("build" in gradle.startParameter.taskNames && testsResults.isNotEmpty()) {
            println("========== Test duration ==========")
            for ((value, key) in testsResults.entries.take(5))
                println("\t$value:\t${key}s")
        }
    }
}

gradle.serviceOf<FlowScope>().always(DoSomething::class.java) { }

var Task.startTime: TimeSource.Monotonic.ValueTimeMark
    get() = extensions.getByName<TimeSource.Monotonic.ValueTimeMark>("starTime")
    set(value) {
        extensions.add("startTime", timeSource.markNow())
    }