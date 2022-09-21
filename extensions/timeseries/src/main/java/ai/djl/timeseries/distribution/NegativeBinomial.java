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
 * <p>Two arguments for this distribution. {@code total_count} non-negative number of negative
 * Bernoulli trials to stop, {@code logits} Event log-odds for probabilities of success
 */
public final class NegativeBinomial extends Distribution {

    private NDArray totalCount;
    private NDArray logits;

    NegativeBinomial(Builder builder) {
        totalCount = builder.distrArgs.get("total_count");
        logits = builder.distrArgs.get("logits");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logProb(NDArray target) {
        NDArray logUnnormalizedProb =
                totalCount.mul(logSigmoid(logits.mul(-1))).add(target.mul(logSigmoid(logits)));

        NDArray logNormalization =
                totalCount
                        .add(target)
                        .gammaln()
                        .mul(-1)
                        .add(target.add(1).gammaln())
                        .add(totalCount.gammaln());
        return logUnnormalizedProb.sub(logNormalization);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sample(int numSamples) {
        NDManager manager = totalCount.getManager();
        NDArray expandedTotalCount =
                numSamples > 0 ? totalCount.expandDims(0).repeat(0, numSamples) : totalCount;
        NDArray expandedLogits =
                numSamples > 0 ? logits.expandDims(0).repeat(0, numSamples) : logits;

        return manager.samplePoisson(manager.sampleGamma(expandedTotalCount, expandedLogits.exp()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return totalCount.mul(logits.exp());
    }

    private NDArray logSigmoid(NDArray x) {
        return x.mul(-1).exp().add(1).getNDArrayInternal().rdiv(1).log();
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
                    distrArgs.contains("total_count"),
                    "NegativeBinomial's args must contain total_count.");
            Preconditions.checkArgument(
                    distrArgs.contains("logits"), "NegativeBinomial's args must contain logits.");
            // We cannot scale using the affine transformation since negative binomial should return
            // integers. Instead we scale the parameters.
            if (scale != null) {
                NDArray logits = distrArgs.get("logits");
                logits.add(scale.log());
                logits.setName("logits");
                distrArgs.remove("logits");
                distrArgs.add(logits);
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
