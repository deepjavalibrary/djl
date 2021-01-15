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
package ai.djl.training.initializer;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * Naive implementation of a truncated normal initializer. Simply samples from a normal distribution
 * and throws away anything outside two standard deviations.
 *
 * @see <a
 *     href="https://en.wikipedia.org/wiki/Truncated_normal_distribution">https://en.wikipedia.org/wiki/Truncated_normal_distribution</a>
 */
@SuppressWarnings("unused")
public class TruncatedNormalInitializer implements Initializer {

    private final float sigma;

    /** Creates an instance of {@code TruncatedNormalInitializer} with a default sigma of 0.01. */
    public TruncatedNormalInitializer() {
        this(0.01f);
    }

    /**
     * Creates a TruncatedNormalInitializer initializer.
     *
     * @param sigma the standard deviation of the truncated normal distribution. Values outside
     *     (-2σ, 2σ) will be rejected.
     */
    public TruncatedNormalInitializer(final float sigma) {
        this.sigma = sigma;
    }

    @Override
    public NDArray initialize(
            final NDManager baseManager, final Shape shape, final DataType dataType) {
        long size = shape.size();
        if (size < 0) {
            throw new IllegalArgumentException("Shape is not determined.");
        }
        // We need to clean up intermediary arrays, so we perform all initialization in our own
        // memory scope.
        NDManager manager = baseManager.newSubManager();

        // We start with an empty array to which we will concat non-rejected samples
        NDArray result = manager.create(new float[] {}, new Shape(0));
        // We keep count of the steps - this should normally take only up to three steps
        // (almost always only one),  we need to stop if we have too many steps as something
        // would be seriously wrong then
        int steps = 0;
        NDArray lowerBound = manager.create(-2f * sigma);
        NDArray upperBound = manager.create(2f * sigma);
        // Repeat until enough samples are within the truncated normal distribution
        while (result.size() < size) {
            // We create more samples than we need, as we have to discard some.
            // 95,45 % of samples are expected to fit, so we create 10% more - that will most
            // likely by enough so we have our result in one go.
            long samplesToCreate = (long) ((size - result.size()) * 1.1);
            // Create normal distribution
            final NDArray normalDistribution =
                    manager.randomNormal(
                            0.0f, sigma, new Shape(samplesToCreate), dataType, manager.getDevice());
            // Create bitmask for all elements that are inside 2σ
            final NDArray larger2Sigma = normalDistribution.gt(lowerBound);
            final NDArray smaller2Sigma = normalDistribution.lt(upperBound);
            final NDArray withinBounds = larger2Sigma.logicalAnd(smaller2Sigma);
            // Select elements that fit criteria
            final NDArray truncatedNormalDistribution = normalDistribution.get(withinBounds);
            // Concat to result
            final NDArray newResult = result.concat(truncatedNormalDistribution);
            result = newResult;
            steps++;
            if (steps > 10) {
                throw new IllegalStateException(
                        "Initialization of truncated normal takes too long - This is incredibly "
                                + "unlikely, something must be seriously wrong.");
            }
        }
        // truncate superfluous values
        result = result.get(new NDIndex().addSliceDim(0, size));
        // reshape to target size
        result = result.reshape(shape);
        result.attach(baseManager);
        manager.close();
        // done!
        return result;
    }
}
