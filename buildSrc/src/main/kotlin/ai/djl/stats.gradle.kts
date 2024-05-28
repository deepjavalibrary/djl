package ai.djl

import org.gradle.api.plugins.ExtraPropertiesExtension
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
        if (/*"build" in gradle.startParameter.taskNames && */parameters.testsResults.isNotEmpty()) {
            println("========== Test duration ========== ")
            for ((key, value) in parameters.testsResults.entries.sortedByDescending { it.key }.take(6))
                // `.inWholeSeconds.seconds` truncate to integer units, without decimals
                println("\t$value:\t${key.inWholeSeconds.seconds}")
        }
    }
}

tasks.test {
    doFirst {
        @OptIn(ExperimentalTime::class)
        startTime = timeSource.markNow()
    }
    doLast {
        @OptIn(ExperimentalTime::class)
        if (state.didWork)
            demoListener.get().parameters.testsResults[timeSource.markNow() - startTime] = project.name
    }
}

@ExperimentalTime
var Task.startTime: TimeSource.Monotonic.ValueTimeMark
    get() = ext.get("startTime") as TimeSource.Monotonic.ValueTimeMark
    set(value) = ext.set("startTime", value)

val Task.ext: ExtraPropertiesExtension
    get() = (this as ExtensionAware).extensions.getByName<ExtraPropertiesExtension>("ext")