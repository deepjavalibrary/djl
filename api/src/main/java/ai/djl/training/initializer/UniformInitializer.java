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

/**
 * {@code UniformInitializer} initializes weights with random values uniformly sampled from a given
 * range.
 */
public class UniformInitializer implements Initializer {
    private float scale;

    /** Creates an instance of {@code UniformInitializer} with a default {@code scale} of 0.07. */
    public UniformInitializer() {
        this.scale = 0.07f;
    }

    /**
     * Initializes a uniform initializer.
     *
     * @param scale the bound on the range of the generated random values. Values are generated from
     *     the range [-`scale`, `scale`]. Default scale is 0.07.
     */
    public UniformInitializer(float scale) {
        this.scale = scale;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType) {
        return manager.randomUniform(-scale, scale, shape, dataType, manager.getDevice());
    }
}
