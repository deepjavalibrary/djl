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

package software.amazon.ai.training.initializer;

import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;

/** Initializes weights with random values uniformly sampled from a given range. */
public class UniformInitializer implements Initializer {
    private float scale;

    public UniformInitializer() {
        this.scale = 0.07f;
    }

    /**
     * Initialize uniform initializer.
     *
     * @param scale float, The bound on the range of the generated random values. Values are
     *     generated from the range [-`scale`, `scale`]. Default scale is 0.07.
     */
    public UniformInitializer(float scale) {
        this.scale = scale;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray initialize(NDManager manager, Shape shape, DataType dataType, Device device) {
        return manager.randomUniform(-scale, scale, shape, dataType, device);
    }
}
