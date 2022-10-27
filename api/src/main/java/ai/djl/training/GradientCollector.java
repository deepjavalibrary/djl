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
 *
 * <p>The typical behavior is to open up a gradient collector during each batch and close it during
 * the end of the batch. In this way, the gradient is reset between batches. If the gradient
 * collector is left open for multiple calls to backwards, the gradients collected are accumulated
 * and added together.
 *
 * <p>Due to limitations in most engines, the gradient collectors are global. This means that only
 * one can be used at a time. If multiple are opened, an error will be thrown.
 */
public interface GradientCollector extends AutoCloseable {

    /**
     * Calculate the gradient w.r.t previously marked variable (head).
     *
     * @param target the target NDArray to calculate the gradient w.r.t head
     */
    void backward(NDArray target);

    /** Sets all the gradients within the engine to zero. */
    void zeroGradients();

    /** {@inheritDoc} */
    @Override
    void close();
}
