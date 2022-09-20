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

/** Represents the distribution of an affinely transformed random variable. */
public class AffineTransformed extends Distribution {

    private Distribution baseDistribution;
    private NDArray loc;
    private NDArray scale;

    /**
     * Construct a new {@code AffineTransformed}
     *
     * <p>This is the distribution of Y = scale * X + loc, where X is a random variable distributed
     * according to {@code baseDistribution}.
     *
     * @param baseDistribution original distribution
     * @param loc translation parameter of the affine transformation
     * @param scale scaling parameter of the affine transformation
     */
    public AffineTransformed(Distribution baseDistribution, NDArray loc, NDArray scale) {
        this.baseDistribution = baseDistribution;
        this.loc = loc == null ? baseDistribution.mean().zerosLike() : loc;
        this.scale = scale == null ? baseDistribution.mean().onesLike() : scale;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logProb(NDArray target) {
        NDArray x = fInv(target);
        NDArray ladj = logAbsDetJac(x);
        NDArray lp = ladj.mul(-1);
        return baseDistribution.logProb(x).add(lp);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sample(int numSamples) {
        NDArray sample = baseDistribution.sample(numSamples);
        return f(sample);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return baseDistribution.mean().mul(scale).add(loc);
    }

    private NDArray f(NDArray x) {
        return x.mul(scale).add(loc);
    }

    private NDArray fInv(NDArray y) {
        return y.sub(loc).div(scale);
    }

    private NDArray logAbsDetJac(NDArray x) {
        return scale.broadcast(x.getShape()).abs().log();
    }
}
