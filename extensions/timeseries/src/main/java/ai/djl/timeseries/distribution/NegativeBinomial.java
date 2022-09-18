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
import ai.djl.ndarray.NDManager;
import ai.djl.util.Preconditions;

/**
 * Negative binomial distribution.
 *
 * <p>The distribution of the number of successes in a sequence of independent Bernoulli trials.
 *
 * <p>Two arguments for this distribution. {@code mu} mean of the distribution, {@code alpha} the
 * inverse number of negative Bernoulli trials to stop
 */
public final class NegativeBinomial extends Distribution {

    private NDArray mu;
    private NDArray alpha;

    NegativeBinomial(Builder builder) {
        mu = builder.distrArgs.get("mu");
        alpha = builder.distrArgs.get("alpha");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logProb(NDArray target) {

        NDArray alphaInv = alpha.getNDArrayInternal().rdiv(1);
        NDArray alphaTimesMu = alpha.mul(mu);

        return target.mul(alphaTimesMu.div(alphaTimesMu.add(1)).log())
                .sub(alphaInv.mul(alphaTimesMu.add(1).log()))
                .add(target.add(alphaInv).gammaln())
                .sub(target.add(1.).gammaln())
                .sub(alphaInv.gammaln());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sample(int numSamples) {
        NDManager manager = mu.getManager();
        NDArray expandedMu = numSamples > 0 ? mu.expandDims(0).repeat(0, numSamples) : mu;
        NDArray expandedAlpha = numSamples > 0 ? alpha.expandDims(0).repeat(0, numSamples) : alpha;

        NDArray r = expandedAlpha.getNDArrayInternal().rdiv(1f);
        NDArray theta = expandedAlpha.mul(expandedMu);
        return manager.samplePoisson(manager.sampleGamma(r, theta));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return mu;
    }

    /**
     * Creates a builder to build a {@code NegativeBinomial}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder to construct a {@code NegativeBinomial}. */
    public static final class Builder extends DistributionBuilder<Builder> {

        /** {@inheritDoc} */
        @Override
        public Distribution build() {
            Preconditions.checkArgument(
                    distrArgs.contains("mu"), "NegativeBinomial's args must contain mu.");
            Preconditions.checkArgument(
                    distrArgs.contains("alpha"), "NegativeBinomial's args must contain alpha.");
            // We cannot scale using the affine transformation since negative binomial should return
            // integers. Instead we scale the parameters.
            if (scale != null) {
                NDArray mu = distrArgs.get("mu");
                mu = mu.mul(scale);
                mu.setName("mu");
                distrArgs.remove("mu");
                distrArgs.add(mu);
            }
            return new NegativeBinomial(this);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }
}
