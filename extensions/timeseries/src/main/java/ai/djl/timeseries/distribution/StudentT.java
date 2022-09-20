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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Preconditions;

/**
 * Student's t-test distribution.
 *
 * <p>Three arguments for this distribution. {@code mu} mean of the distribution, {@code sigma} the
 * standard deviations (scale), {@code nu} degrees of freedom.
 */
public class StudentT extends Distribution {

    private NDArray mu;
    private NDArray sigma;
    private NDArray nu;

    StudentT(Builder builder) {
        mu = builder.distrArgs.get("mu");
        sigma = builder.distrArgs.get("sigma");
        nu = builder.distrArgs.get("nu");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logProb(NDArray target) {
        NDArray nup1Half = nu.add(1.).div(2.);
        NDArray part1 = nu.getNDArrayInternal().rdiv(1.).mul(target.sub(mu).div(sigma).square());

        NDArray z =
                nup1Half.gammaln()
                        .sub(nu.div(2.).gammaln())
                        .sub(nu.mul(Math.PI).log().mul(0.5))
                        .sub(sigma.log());

        return z.sub(nup1Half.mul(part1.add(1.).log()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sample(int numSamples) {
        NDManager manager = mu.getManager();
        NDArray expandedMu = numSamples > 0 ? mu.expandDims(0).repeat(0, numSamples) : mu;
        NDArray expandedSigma = numSamples > 0 ? sigma.expandDims(0).repeat(0, numSamples) : sigma;
        NDArray expandedNu = numSamples > 0 ? nu.expandDims(0).repeat(0, numSamples) : nu;

        NDArray gammas =
                manager.sampleGamma(
                        expandedNu.div(2.),
                        expandedNu.mul(expandedSigma.square()).getNDArrayInternal().rdiv(2.));
        return manager.sampleNormal(expandedMu, gammas.sqrt().getNDArrayInternal().rdiv(1.));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return NDArrays.where(nu.gt(1.0), mu, mu.getManager().full(mu.getShape(), Float.NaN));
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
                    distrArgs.contains("mu"), "StudentTl's args must contain mu.");
            Preconditions.checkArgument(
                    distrArgs.contains("sigma"), "StudentTl's args must contain sigma.");
            Preconditions.checkArgument(
                    distrArgs.contains("nu"), "StudentTl's args must contain nu.");
            StudentT baseDistr = new StudentT(this);
            if (scale == null && loc == null) {
                return baseDistr;
            }
            return new AffineTransformed(baseDistr, loc, scale);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }
    }
}
