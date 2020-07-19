/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/** Initializer that generates tensors with constant values. */
public class ConstantInitializer implements Initializer {

    private float value;

    /**
     * Creates a Constant Initializer.
     *
     * @param value the value to fill
     */
    public ConstantInitializer(float value) {
        this.value = value;
    }

    /**
     * Initializes a single {@link NDArray}.
     *
     * @param manager the {@link NDManager} to create the new {@link NDArray} in
     * @param shape the {@link Shape} for the new NDArray
     * @param dataType the {@link DataType} for the new NDArray
     * @return the {@link NDArray} initialized with the manager and shape
     */
    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        return manager.full(shape, value, dataType);
    }
}
