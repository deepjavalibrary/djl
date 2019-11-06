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

import ai.djl.ndarray.NDArray;

/**
 * An interface that provides a mechanism to collect gradients during training.
 *
 * <p>The {@code GradientCollector} should be opened with a try-with-resources. All operations
 * performed within the try-with-resources are recorded and the variables marked. When {@link
 * #backward(NDArray) backward function} is called, gradients are collected w.r.t previously marked
 * variables.
 */
public interface GradientCollector extends AutoCloseable {

    /**
     * Calculate the gradient w.r.t previously marked variable (head).
     *
     * @param target the target NDArray to calculate the gradient w.r.t head
     */
    void backward(NDArray target);

    /** {@inheritDoc} */
    @Override
    void close();
}
