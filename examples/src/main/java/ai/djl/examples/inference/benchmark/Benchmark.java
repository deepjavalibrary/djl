/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.examples.inference.benchmark;

import ai.djl.ModelException;
import ai.djl.examples.inference.benchmark.util.AbstractBenchmark;
import ai.djl.examples.inference.benchmark.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.listener.MemoryTrainingListener;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Arrays;

public final class Benchmark extends AbstractBenchmark {

    public static void main(String[] args) {
        if (Arrays.asList(args).contains("-t")) {
            MultithreadedBenchmark.main(args);
            return;
        }
        if (new Benchmark().runBenchmark(args)) {
            System.exit(0); // NOPMD
        }
        System.exit(-1); // NOPMD
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public Object predict(Arguments arguments, Metrics metrics, int iteration)
            throws IOException, ModelException, TranslateException, ClassNotFoundException {
        Object inputData = arguments.getInputData();
        try (ZooModel<?, ?> model = loadModel(arguments, metrics)) {
            Object predictResult = null;
            try (Predictor predictor = model.newPredictor()) {
                predictor.setMetrics(metrics); // Let predictor collect metrics

                for (int i = 0; i < iteration; ++i) {
                    predictResult = predictor.predict(inputData);

                    progressBar.update(i);
                    MemoryTrainingListener.collectMemoryInfo(metrics);
                }
            }
            return predictResult;
        }
    }
}
