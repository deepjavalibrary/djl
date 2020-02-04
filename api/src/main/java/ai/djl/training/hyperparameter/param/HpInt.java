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
package ai.djl.training.hyperparameter.param;

import ai.djl.util.RandomUtils;

/** A {@link Hyperparameter} for an integer. */
public class HpInt extends Hyperparameter<Integer> {

    int lower;
    int upper;

    /**
     * Constructs a {@link HpInt}.
     *
     * @param name the name of the hyperparameter
     * @param lower the lower bound (inclusive)
     * @param upper the upper bound (exclusive)
     */
    public HpInt(String name, int lower, int upper) {
        super(name);
        this.lower = lower;
        this.upper = upper;
    }

    /** {@inheritDoc} */
    @Override
    public Integer random() {
        int range = upper - lower;
        return RandomUtils.nextInt(range) + lower;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "HPInt{" + "lower=" + lower + ", upper=" + upper + ", name='" + name + '\'' + '}';
    }
}
