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
package ai.djl.zero;

/**
 * Describes the speed/accuracy tradeoff.
 *
 * <p>In deep learning, it is often possible to improve the accuracy of a model by using a larger
 * model. However, this then results in slower latency and worse throughput. So, there is a tradeoff
 * between the choices of speed and accuracy.
 */
public enum Performance {
    /** Fast prioritizes speed over accuracy. */
    FAST,

    /** Balanced has a more even tradeoff of speed and accuracy. */
    BALANCED,

    /** Accurate prioritizes accuracy over speed. */
    ACCURATE;

    /**
     * Returns the value matching this.
     *
     * @param fast the value to return if this is fast
     * @param balanced the value to return if this is balanced
     * @param accurate the value to return if this is accurate
     * @param <T> the value type
     * @return the value matching this
     */
    public <T> T switchPerformance(T fast, T balanced, T accurate) {
        switch (this) {
            case FAST:
                return fast;
            case BALANCED:
                return balanced;
            case ACCURATE:
                return accurate;
            default:
                throw new IllegalArgumentException("Unknown performance");
        }
    }
}
