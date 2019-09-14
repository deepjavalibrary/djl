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
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.translate.TrainTranslator;
import software.amazon.ai.translate.TranslateException;
import software.amazon.ai.translate.TranslatorContext;

public interface Trainer<I, L, O> extends AutoCloseable {

    TrainTranslator<I, L, O> getTranslator();

    TranslatorContext getPreprocessContext();

    default Iterable<Batch> iterateDataset(Dataset<I, L> dataset) throws IOException {
        return dataset.getData(this);
    }

    /**
     * Predicts the method used for inference.
     *
     * @param input Input follows the inputObject
     * @return The Output object defined by user
     * @throws TranslateException if an error occurs during prediction
     */
    default O predict(I input) throws TranslateException {
        return predict(Collections.singletonList(input)).get(0);
    }

    /**
     * Predicts the method used for inference.
     *
     * @param input Inputs follows the inputObject
     * @return The Output objects defined by user
     * @throws TranslateException if an error occurs during prediction
     */
    List<O> predict(List<I> input) throws TranslateException;

    void step();

    /**
     * Attaches a Metrics param to use for benchmark.
     *
     * @param metrics the Metrics class
     */
    void setMetrics(Metrics metrics);

    NDManager getManager();

    Optional<Integer> getSeed();

    void setSeed(int seed);

    ModelSaver getModelSaver();

    void setModelSaver(ModelSaver modelSaver);

    void checkpoint();

    /** {@inheritDoc} */
    @Override
    void close();
}
