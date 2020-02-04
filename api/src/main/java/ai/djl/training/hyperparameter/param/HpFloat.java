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

/** A {@link Hyperparameter} for a float. */
public class HpFloat extends Hyperparameter<Float> {

    private float lower;
    private float upper;
    private boolean log;

    /**
     * Constructs a {@link HpFloat}.
     *
     * @param name the name of the hyperparameter
     * @param lower the lower bound (inclusive)
     * @param upper the upper bound (exclusive)
     * @param log whether to use log space. This is useful if your bounds cover different orders of
     *     magnitude (e.g. 1E-5 to 1E-2) instead of same magnitude (e.g. 2 to 5).
     */
    public HpFloat(String name, float lower, float upper, boolean log) {
        super(name);
        this.log = log;
        this.lower = lower;
        this.upper = upper;
    }

    /** {@inheritDoc} */
    @Override
    public Float random() {
        if (log) {
            float logLower = (float) Math.log(lower);
            float logUpper = (float) Math.log(upper);
            return (float) Math.exp(RandomUtils.nextFloat(logLower, logUpper));
        } else {
            return RandomUtils.nextFloat(lower, upper);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "HPReal{"
                + "lower="
                + lower
                + ", upper="
                + upper
                + ", log="
                + log
                + ", name='"
                + name
                + '\''
                + '}';
    }
}
