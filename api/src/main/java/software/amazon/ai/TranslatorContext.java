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
package software.amazon.ai;

import software.amazon.ai.metric.Metrics;
import software.amazon.ai.ndarray.NDFactory;

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
     * @return {@link Model}
     */
    Model getModel();

    /**
     * Returns the context information.
     *
     * @return {@link Context}
     */
    Context getContext();

    /**
     * Returns the {@link NDFactory} to create {@link software.amazon.ai.ndarray.NDArray}.
     *
     * @return {@link NDFactory}
     */
    NDFactory getNDFactory();

    /**
     * Returns the Metric tool to do benchmark.
     *
     * @return {@link Metrics}
     */
    Metrics getMetrics();

    @Override
    void close();
}
