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
package ai.djl.modality.cv.transform;

import ai.djl.ndarray.NDArray;
import ai.djl.translate.Transform;
import ai.djl.util.RandomUtils;

import java.util.Random;

/**
 * A {@link Transform} that randomly flip the input image left to right with a probability of 0.5.
 */
public class RandomFlipLeftRight implements Transform {
    Integer seed;

    /** Creates a new instance of {@code RandomFlipLeftRight}. */
    public RandomFlipLeftRight() {}

    /**
     * Creates a new instance of {@code RandomFlipLeftRight} with the given seed.
     *
     * @param seed the value of the seed
     */
    public RandomFlipLeftRight(int seed) {
        this.seed = seed;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transform(NDArray array) {
        Random rnd = (seed != null) ? new Random(seed) : RandomUtils.RANDOM;
        if (rnd.nextFloat() > 0.5) {
            array.flip(1);
        }
        return array;
    }
}
