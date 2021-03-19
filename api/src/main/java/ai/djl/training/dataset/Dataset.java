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
package ai.djl.training.dataset;

import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.concurrent.ExecutorService;

/**
 * An interface to represent a set of sample data/label pairs to train a model.
 *
 * @see <a href="http://docs.djl.ai/docs/dataset.html">The guide on datasets</a>
 */
public interface Dataset {

    /**
     * Fetches an iterator that can iterate through the {@link Dataset}.
     *
     * @param manager the dataset to iterate through
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    Iterable<Batch> getData(NDManager manager) throws IOException, TranslateException;

    /**
     * Fetches an iterator that can iterate through the {@link Dataset} with multiple threads.
     *
     * @param manager the dataset to iterate through
     * @param executorService the executorService to use for multi-threading
     * @return an {@link Iterable} of {@link Batch} that contains batches of data from the dataset
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    default Iterable<Batch> getData(NDManager manager, ExecutorService executorService)
            throws IOException, TranslateException {
        return getData(manager);
    }

    /**
     * Prepares the dataset for use.
     *
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    default void prepare() throws IOException, TranslateException {
        prepare(null);
    }

    /**
     * Prepares the dataset for use with tracked progress.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    void prepare(Progress progress) throws IOException, TranslateException;

    /** An enum that indicates the mode - training, test or validation. */
    enum Usage {
        TRAIN,
        TEST,
        VALIDATION
    }
}
