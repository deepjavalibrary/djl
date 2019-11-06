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
package ai.djl.training;

import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.TrainingMetric;

public interface Trainer extends AutoCloseable {

    void initialize(Shape[] shapes);

    default Iterable<Batch> iterateDataset(Dataset dataset) {
        return dataset.getData(getManager());
    }

    GradientCollector newGradientCollector();

    void trainBatch(Batch batch);

    NDList forward(NDList input);

    void validateBatch(Batch batch);

    /** Makes one step of parameter update. */
    void step();

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    void setTrainingListener(TrainingListener listener);

    void resetTrainingMetrics();

    Loss getLoss();

    Loss getValidationLoss();

    <T extends TrainingMetric> T getTrainingMetric(Class<T> clazz);

    <T extends TrainingMetric> T getValidationMetric(Class<T> clazz);

    NDManager getManager();

    /** {@inheritDoc} */
    @Override
    void close();
}
