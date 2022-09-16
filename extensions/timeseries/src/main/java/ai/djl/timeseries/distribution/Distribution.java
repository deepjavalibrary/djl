/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.distribution;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;

/** An abstract class representing probability distribution. */
public abstract class Distribution {

    /**
     * Compute the log of the probability density/mass function evaluated at target.
     *
     * @param target {@link NDArray} of shape (*batch_shape, *event_shape)
     * @return Tensor of shape (batch_shape) containing the probability log-density for each event
     *     in target
     */
    public abstract NDArray logProb(NDArray target);

    /**
     * Draw samples from the distribution.
     *
     * <p>The first dimension of the output will be numSamples.
     *
     * @param numSamples Number of samples to be drawn
     * @return a {@link NDArray} has shape (num_samples, *batch_shape, *target_shape)
     */
    public abstract NDArray sample(int numSamples);

    /**
     * Return the mean of the distribution.
     *
     * @return the mean of the distribution
     */
    public abstract NDArray mean();

    /**
     * A builder to extend for all classes extend the {@link Distribution}.
     *
     * @param <T> the concrete builder type
     */
    public abstract static class DistributionBuilder<T extends DistributionBuilder<T>> {
        protected NDList distrArgs;
        protected NDArray scale;
        protected NDArray loc;

        /**
         * Set the appropriate arguments for the probability distribution.
         *
         * @param distrArgs a {@link NDList} containing distribution args named after the parameter
         *     name
         * @return this builder
         */
        public T setDistrArgs(NDList distrArgs) {
            this.distrArgs = distrArgs;
            return self();
        }

        /**
         * Set the affine scale for the probability distribution.
         *
         * @param scale the affine scale
         * @return this builder
         */
        public T optScale(NDArray scale) {
            this.scale = scale;
            return self();
        }

        /**
         * Set the affine location of the probability.
         *
         * @param loc the affine location
         * @return this builder
         */
        public T optLoc(NDArray loc) {
            this.loc = loc;
            return self();
        }

        /**
         * Build a {@code Distribution}.
         *
         * @return the {@code Distribution}
         */
        public abstract Distribution build();

        protected abstract T self();
    }
}
