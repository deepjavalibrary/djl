plugins {
    base
    `jacoco-report-aggregation`
}

repositories {
    mavenCentral()
}

dependencies {
    jacocoAggregation(projects.api)
    jacocoAggregation(projects.basicdataset)
    jacocoAggregation(projects.engines.llama)
    jacocoAggregation(projects.engines.ml.xgboost)
    jacocoAggregation(projects.engines.ml.lightgbm)
    jacocoAggregation(projects.engines.mxnet.mxnetEngine)
    jacocoAggregation(projects.engines.mxnet.mxnetModelZoo)
    jacocoAggregation(projects.engines.mxnet.native)
    jacocoAggregation(projects.engines.onnxruntime.onnxruntimeAndroid)
    jacocoAggregation(projects.engines.onnxruntime.onnxruntimeEngine)
    jacocoAggregation(projects.engines.pytorch.pytorchEngine)
    jacocoAggregation(projects.engines.pytorch.pytorchJni)
    jacocoAggregation(projects.engines.pytorch.pytorchModelZoo)
    jacocoAggregation(projects.engines.pytorch.pytorchNative)
    jacocoAggregation(projects.engines.tensorflow.tensorflowApi)
    jacocoAggregation(projects.engines.tensorflow.tensorflowEngine)
    jacocoAggregation(projects.engines.tensorflow.tensorflowModelZoo)
    jacocoAggregation(projects.engines.tensorflow.tensorflowNative)
    jacocoAggregation(projects.examples)
    jacocoAggregation(projects.extension.audio)
    jacocoAggregation(projects.extension.fasttext)
    jacocoAggregation(projects.extension.hadoop)
    jacocoAggregation(projects.extension.opencv)
    jacocoAggregation(projects.extension.sentencepiece)
    jacocoAggregation(projects.extension.tokenizers)
    jacocoAggregation(projects.extension.tablesaw)
    jacocoAggregation(projects.extension.timeseries)
    if (JavaVersion.current() < JavaVersion.VERSION_19)
        jacocoAggregation(projects.extension.spark)
    jacocoAggregation(projects.integration)
    //    jacocoAggregation(projects.modelzoo)
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
                exclude("org/tensorflow/lite/**",
                        "ai/djl/integration/**",
                        "ai/djl/examples/**")
            }
        }))
    }
    check {
        dependsOn(testCodeCoverageReport)
    }
}
