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
package software.amazon.ai.training;

import java.io.IOException;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.optimizer.Optimizer;

public interface Trainer extends AutoCloseable {

    default Iterable<Batch> iterateDataset(Dataset dataset) throws IOException {
        return dataset.getData();
    }

    GradientCollector newGradientCollector();

    /**
     * An internal helper to get the Engine specific implementation for parameter store.
     *
     * @param optimizer The optimizer that defines how to update parameters
     * @return {@link ParameterServer} object
     */
    ParameterServer newParameterServer(Optimizer optimizer);

    NDList forward(NDList input);

    /** Makes one step of parameter update. */
    void step();

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    NDManager getManager();

    /** {@inheritDoc} */
    @Override
    void close();
}
