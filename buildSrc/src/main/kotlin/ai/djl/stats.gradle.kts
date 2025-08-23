package ai.djl

import org.gradle.kotlin.dsl.support.serviceOf
import org.gradle.tooling.events.FinishEvent
import org.gradle.tooling.events.OperationCompletionListener
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

plugins {
    `java-library`
}

@ExperimentalTime
val timeSource = TimeSource.Monotonic
val testsResults = mutableMapOf<Duration, String>()

val demoListener = gradle.sharedServices.registerIfAbsent("demoListener ", StatisticsService::class) {
    parameters.testsResults = testsResults
}

gradle.taskGraph.whenReady {
    gradle.serviceOf<BuildEventsListenerRegistry>().onTaskCompletion(demoListener)
}

abstract class StatisticsService : BuildService<StatisticsService.Parameters>,
    OperationCompletionListener, AutoCloseable {

    interface Parameters : BuildServiceParameters {
        var testsResults: MutableMap<Duration, String>
    }

    override fun onFinish(event: FinishEvent) {}

    override fun close() {
        if (parameters.testsResults.isNotEmpty()) {
            println("========== Test duration ========== ")
            for ((key, value) in parameters.testsResults.entries.sortedByDescending { it.key }.take(6)) {
                // `.inWholeSeconds.seconds` truncate to integer units, without decimals
                println("\t$value:\t${key.inWholeSeconds.seconds}")
            }
        }
    }
}

tasks.test {
    @OptIn(ExperimentalTime::class)
    val timeSource = timeSource
    val projectName = project.name
    val demoListener = demoListener
    val ext = project.ext
    doFirst {
        @OptIn(ExperimentalTime::class)
        ext.set("startTime", timeSource.markNow())
    }
    doLast {
        @OptIn(ExperimentalTime::class)
        if (state.didWork) {
            val t = ext.get("startTime") as TimeSource.Monotonic.ValueTimeMark
            demoListener.get().parameters.testsResults[timeSource.markNow() - t] = projectName
        }
    }
}
