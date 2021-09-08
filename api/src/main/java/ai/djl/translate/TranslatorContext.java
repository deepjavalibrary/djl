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
package ai.djl.translate;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;

/**
 * The {@code TranslatorContext} interface provides a toolkit for pre-processing and postprocessing
 * functionality.
 *
 * <p>You can use this in {@link Translator} to get Model information and create an NDArray
 */
public interface TranslatorContext extends AutoCloseable {

    /**
     * Returns the {@link Model} object to understand the input/output.
     *
     * @return the {@link Model}
     */
    Model getModel();

    /**
     * Returns the {@link NDManager} to create {@link NDArray}.
     *
     * @return the {@link NDManager}
     */
    NDManager getNDManager();

    /**
     * Returns the Predictor's {@link NDManager}.
     *
     * @return the Predictor's {@link NDManager}
     */
    NDManager getPredictorManager();

    /**
     * Returns the block from the {@code TranslatorContext}.
     *
     * @return the block from the {@code TranslatorContext}
     */
    Block getBlock();

    /**
     * Returns the Metric tool to do benchmark.
     *
     * @return the {@link Metrics}
     */
    Metrics getMetrics();

    /**
     * Returns value of attached key-value pair to context.
     *
     * @param key key of attached value
     * @return the object stored in relevant map
     */
    Object getAttachment(String key);

    /**
     * Set a key-value pair of attachments.
     *
     * @param key key of attached value
     * @param value value assosicated with key
     */
    void setAttachment(String key, Object value);

    /** {@inheritDoc} */
    @Override
    void close();
}
