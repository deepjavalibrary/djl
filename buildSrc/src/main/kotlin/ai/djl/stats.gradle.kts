package ai.djl

import org.gradle.api.plugins.ExtraPropertiesExtension
import org.gradle.kotlin.dsl.support.serviceOf
import org.gradle.tooling.events.FinishEvent
import org.gradle.tooling.events.OperationCompletionListener
import kotlin.time.Duration
import kotlin.time.ExperimentalTime
import kotlin.time.TimeSource

plugins {
    `java-library`
}

@ExperimentalTime
val timeSource = TimeSource.Monotonic
//val testsResults = TreeMap<Duration, String>(Comparator.reverseOrder())
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
        //        if (/*"build" in gradle.startParameter.taskNames && */parameters.testsResults.isNotEmpty()) {
        println("========== Test duration ========== " + parameters.testsResults.size)
        for ((key, value) in parameters.testsResults.entries.sortedByDescending { it.key }.take(6))
            println("\t$value:\t${key}s")
        //        }
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
//            testsResults[startTime - timeSource.markNow()] = project.name
            demoListener.get().parameters.testsResults[timeSource.markNow() - startTime] = project.name
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
    get() = ext.get("startTime") as TimeSource.Monotonic.ValueTimeMark
    set(value) = ext.set("startTime", value)

val Task.ext: ExtraPropertiesExtension
    get() = (this as ExtensionAware).extensions.getByName<ExtraPropertiesExtension>("ext")