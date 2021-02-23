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
package ai.djl.training.initializer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

/**
 * An interface representing an initialization method.
 *
 * <p>Used to initialize the {@link NDArray} parameters stored within a {@link Block}.
 *
 * @see <a
 *     href="https://d2l.djl.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html">The
 *     D2L chapter on numerical stability and initialization</a>
 */
public interface Initializer {

    Initializer ZEROS = (m, s, t) -> m.zeros(s, t, m.getDevice());
    Initializer ONES = (m, s, t) -> m.ones(s, t, m.getDevice());

    /**
     * Initializes a single {@link NDArray}.
     *
     * @param manager the {@link NDManager} to create the new {@link NDArray} in
     * @param shape the {@link Shape} for the new NDArray
     * @param dataType the {@link DataType} for the new NDArray
     * @return the {@link NDArray} initialized with the manager and shape
     */
    NDArray initialize(NDManager manager, Shape shape, DataType dataType);
}
