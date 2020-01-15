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
package ai.djl.training.listener;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Evaluator;

/**
 * {@link TrainingListener} that records evaluator results in metrics.
 *
 * <p>The training and validation evaluators are saved as metrics with name ("train_" +
 * evaluator.getName()) and ("validate_" + evaluator.getName()). The validation evaluators are also
 * saved as model properties with the evaluator name.
 */
public class EvaluatorMetricsTrainingListener implements TrainingListener {

    /** {@inheritDoc} */
    @Override
    public void onEpoch(Trainer trainer) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        for (Evaluator evaluator : trainer.getTrainingEvaluators()) {
            metrics.addMetric("train_" + evaluator.getName(), evaluator.getValue());
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onValidationBatch(Trainer trainer) {
        Metrics metrics = trainer.getMetrics();
        for (Evaluator evaluator : trainer.getValidationEvaluators()) {
            metrics.addMetric("validate_" + evaluator.getName(), evaluator.getValue());
        }
    }

    /** {@inheritDoc} */
    @Override
    public void onTrainingBegin(Trainer trainer) {}

    /** {@inheritDoc} */
    @Override
    public void onTrainingEnd(Trainer trainer) {
        Model model = trainer.getModel();
        Metrics metrics = trainer.getMetrics();
        for (Evaluator evaluator : trainer.getValidationEvaluators()) {
            float value =
                    metrics.latestMetric("validate_" + evaluator.getName()).getValue().floatValue();
            model.setProperty(evaluator.getName(), String.format("%.5f", value));
        }
    }
}
