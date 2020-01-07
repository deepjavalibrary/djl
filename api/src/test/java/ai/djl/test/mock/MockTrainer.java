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

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingListener;
import ai.djl.training.dataset.Batch;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.training.loss.Loss;
import java.util.List;

public class MockTrainer implements Trainer {

    /** {@inheritDoc} */
    @Override
    public void initialize(Shape... shapes) {}

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        return new GradientCollector() {

            /** {@inheritDoc} */
            @Override
            public void backward(NDArray target) {}

            /** {@inheritDoc} */
            @Override
            public void close() {}
        };
    }

    /** {@inheritDoc} */
    @Override
    public void step() {}

    /** {@inheritDoc} */
    @Override
    public Metrics getMetrics() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void trainBatch(Batch batch) {}

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList input) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void validateBatch(Batch batch) {}

    /** {@inheritDoc} */
    @Override
    public void setMetrics(Metrics metrics) {}

    /** {@inheritDoc} */
    @Override
    public void setTrainingListener(TrainingListener listener) {}

    /** {@inheritDoc} */
    @Override
    public void resetEvaluators() {}

    /** {@inheritDoc} */
    @Override
    public Loss getLoss() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Loss getValidationLoss() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Model getModel() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getTrainingEvaluators() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <T extends Evaluator> T getTrainingEvaluator(Class<T> clazz) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<Evaluator> getValidationEvaluators() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <T extends Evaluator> T getValidationEvaluator(Class<T> clazz) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
