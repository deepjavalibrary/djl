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
package ai.djl.test.mock;

import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingListener;
import ai.djl.training.dataset.Batch;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.TrainingMetric;

public class MockTrainer implements Trainer {

    @Override
    public void initialize(Shape[] shapes) {}

    @Override
    public GradientCollector newGradientCollector() {
        return new GradientCollector() {

            @Override
            public void backward(NDArray target) {}

            @Override
            public void close() {}
        };
    }

    @Override
    public void step() {}

    @Override
    public void train(Batch batch) {}

    @Override
    public NDList forward(NDList input) {
        return null;
    }

    @Override
    public void validate(Batch batch) {}

    @Override
    public void setMetrics(Metrics metrics) {}

    @Override
    public void setTrainingListener(TrainingListener listener) {}

    @Override
    public void resetTrainingMetrics() {}

    @Override
    public Loss getLoss() {
        return null;
    }

    @Override
    public Loss getValidationLoss() {
        return null;
    }

    @Override
    public <T extends TrainingMetric> T getTrainingMetric(Class<T> clazz) {
        return null;
    }

    @Override
    public <T extends TrainingMetric> T getValidationMetric(Class<T> clazz) {
        return null;
    }

    @Override
    public NDManager getManager() {
        return null;
    }

    @Override
    public void close() {}
}
