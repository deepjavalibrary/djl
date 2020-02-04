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
package ai.djl.examples.training.util;

import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.listener.EvaluatorTrainingListener;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ExampleTrainingResult {

    private String lossName;
    Map<String, Float> evaluations;

    public ExampleTrainingResult(Trainer trainer) {
        lossName = trainer.getLoss().getName();
        Metrics metrics = trainer.getMetrics();
        evaluations = new ConcurrentHashMap<>();
        for (Evaluator evaluator : trainer.getEvaluators()) {
            float value =
                    metrics.latestMetric(
                                    EvaluatorTrainingListener.metricName(
                                            evaluator, EvaluatorTrainingListener.VALIDATE_EPOCH))
                            .getValue()
                            .floatValue();
            evaluations.put(evaluator.getName(), value);
        }
    }

    public float getEvaluation(String name) {
        return evaluations.get(name);
    }

    public float getLoss() {
        return evaluations.get(lossName);
    }
}
