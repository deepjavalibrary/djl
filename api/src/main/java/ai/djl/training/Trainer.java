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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.metrics.TrainingMetrics;
import java.io.IOException;
import java.util.List;

public interface Trainer extends AutoCloseable {

    void initialize(DataDesc[] inputDescriptor);

    default Iterable<Batch> iterateDataset(Dataset dataset) throws IOException {
        return dataset.getData(getManager());
    }

    GradientCollector newGradientCollector();

    NDList forward(NDList input);

    void validate(NDList input, NDList label);

    NDArray loss(NDList labels, NDList preds);

    /** Makes one step of parameter update. */
    void step();

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    void resetTrainingMetrics();

    float getLoss();

    float getValidationLoss();

    List<TrainingMetrics> getTrainingMetrics();

    List<TrainingMetrics> getValidateMetrics();

    NDManager getManager();

    /** {@inheritDoc} */
    @Override
    void close();
}
