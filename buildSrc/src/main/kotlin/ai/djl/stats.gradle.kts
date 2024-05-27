package ai.djl

import org.gradle.tooling.events.OperationCompletionListener
import java.util.*
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

plugins {
    `java-library`
}

@ExperimentalTime
val timeSource = TimeSource.Monotonic
val testsResults = TreeMap<Duration, String>(Comparator.reverseOrder())

tasks.test {
    doFirst {
        @OptIn(ExperimentalTime::class)
        startTime = timeSource.markNow()
    }
    doLast {
        @OptIn(ExperimentalTime::class)
        if (state.didWork)
            testsResults[startTime - timeSource.markNow()] = project.name
    }
}


abstract class StatisticsService : BuildService<StatisticsService.Parameters>,
                                   OperationCompletionListener, AutoCloseable {

    interface Parameters : BuildServiceParameters {
        var testsResults: TreeMap<Duration, String>
    }

    override fun close() {
        if ("build" in gradle.startParameter.taskNames && parameters.testsResults.isNotEmpty()) {
            println("========== Test duration ==========")
            for ((value, key) in parameters.testsResults.entries.take(5))
                println("\t$value:\t${key}s")
        }
    }
}

//gradle.buildFinished {
//    if ("build" in gradle!!.startParameter.taskNames && testsResults.isNotEmpty()) {
//        println("========== Test duration ==========")
//        for ((value, key) in testsResults.entries.take(5))
//            println("\t$value:\t${key}s")
//    }
//}

@ExperimentalTime
var Task.startTime: TimeSource.Monotonic.ValueTimeMark
    get() = extensions.getByName<TimeSource.Monotonic.ValueTimeMark>("starTime")
    set(value) {
        extensions.add("startTime", value)
    }