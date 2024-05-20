plugins {
    base
    `jacoco-report-aggregation`
}

repositories {
    mavenCentral()
}

dependencies {
    jacocoAggregation(project(":api"))
    jacocoAggregation(project(":basicdataset"))
    jacocoAggregation(project(":engines:llama"))
    jacocoAggregation(project(":engines:ml:xgboost"))
    jacocoAggregation(project(":engines:ml:lightgbm"))
    jacocoAggregation(project(":engines:mxnet:mxnet-engine"))
    jacocoAggregation(project(":engines:mxnet:mxnet-model-zoo"))
    jacocoAggregation(project(":engines:mxnet:native"))
    jacocoAggregation(project(":engines:onnxruntime:onnxruntime-android"))
    jacocoAggregation(project(":engines:onnxruntime:onnxruntime-engine"))
    jacocoAggregation(project(":engines:pytorch:pytorch-engine"))
    jacocoAggregation(project(":engines:pytorch:pytorch-jni"))
    jacocoAggregation(project(":engines:pytorch:pytorch-model-zoo"))
    jacocoAggregation(project(":engines:pytorch:pytorch-native"))
    jacocoAggregation(project(":engines:tensorflow:tensorflow-api"))
    jacocoAggregation(project(":engines:tensorflow:tensorflow-engine"))
    jacocoAggregation(project(":engines:tensorflow:tensorflow-model-zoo"))
    jacocoAggregation(project(":engines:tensorflow:tensorflow-native"))
    jacocoAggregation(project(":examples"))
    jacocoAggregation(project(":extensions:audio"))
    jacocoAggregation(project(":extensions:fasttext"))
    jacocoAggregation(project(":extensions:hadoop"))
    jacocoAggregation(project(":extensions:opencv"))
    jacocoAggregation(project(":extensions:sentencepiece"))
    jacocoAggregation(project(":extensions:tokenizers"))
    jacocoAggregation(project(":extensions:tablesaw"))
    jacocoAggregation(project(":extensions:timeseries"))
    if (JavaVersion.current() < JavaVersion.VERSION_19)
        jacocoAggregation(project(":extensions:spark"))
    jacocoAggregation(project(":integration"))
    jacocoAggregation(project(":model-zoo"))
}

reporting {
    reports {
        register<JacocoCoverageReport>("testCodeCoverageReport") {
            testType = TestSuiteType.UNIT_TEST
        }
    }
}

tasks {
    val testCodeCoverageReport by getting(JacocoReport::class) {
        classDirectories.setFrom(files(classDirectories.files.map {
            fileTree(it) {
                exclude(
                    "ai/djl/integration/**",
                    "ai/djl/examples/**"
                )
            }
        }))
    }
    check {
        dependsOn(testCodeCoverageReport)
    }
}
