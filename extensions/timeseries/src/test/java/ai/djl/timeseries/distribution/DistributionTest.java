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
import ai.djl.ndarray.NDManager;
import ai.djl.testing.Assertions;
import ai.djl.testing.TestRequirements;

import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class DistributionTest {

    @BeforeClass
    public void setUp() {
        // TODO: Remove this once we support PyTorch support for timeseries extension
        TestRequirements.notArm();
    }

    @Test
    public void testNegativeBinomial() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray totalCount = manager.create(new float[] {5f, 1f});
            NDArray logits = manager.create(new float[] {0.1f, 2f});
            totalCount.setName("total_count");
            logits.setName("logits");
            Distribution negativeBinomialDistribution =
                    NegativeBinomial.builder().setDistrArgs(new NDList(totalCount, logits)).build();

            NDArray expectedValues = manager.create(new float[] {-2.3027f, -2.2539f});
            NDArray actualValues = negativeBinomialDistribution.logProb(manager.create(new float[] {2f, 1f}));
            Assertions.assertAlmostEquals(actualValues, expectedValues);

            NDArray samplesMean = negativeBinomialDistribution.sample(100000).mean(new int[] {0});
            Assertions.assertAlmostEquals(samplesMean, negativeBinomialDistribution.mean(), 2e-2f, 2e-2f);
        }
    }

    @Test
    public void testStudentT() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray mu = manager.create(new float[] {1000f, -1000f});
            NDArray sigma = manager.create(new float[] {1f, 2f});
            NDArray nu = manager.create(new float[] {4.2f, 3f});
            mu.setName("mu");
            sigma.setName("sigma");
            nu.setName("nu");
            Distribution studentTDistribution =
                    StudentT.builder().setDistrArgs(new NDList(mu, sigma, nu)).build();

            NDArray expectedValues = manager.create(new float[] {-0.9779f, -1.6940f});
            NDArray actualValues = studentTDistribution.logProb(manager.create(new float[] {1000f, -1000f}));
            Assertions.assertAlmostEquals(actualValues, expectedValues);

            NDArray samplesMean = studentTDistribution.sample(100000).mean(new int[] {0});
            Assertions.assertAlmostEquals(samplesMean, studentTDistribution.mean(), 2e-2f, 2e-2f);
        }
    }
}
