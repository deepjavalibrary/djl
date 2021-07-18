/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests.training;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.ElasticNetWeightDecay;
import ai.djl.training.loss.L1WeightDecay;
import ai.djl.training.loss.L2WeightDecay;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

public class WeightDecayTest {

    @Test
    public void l1DecayTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray parameters1 = manager.create(new float[] {-1, -2, 3, 4, 5}); // 15
            NDArray parameters2 = manager.create(new float[] {-1, -1, -1, -1, -1}); // 5
            // Not used
            NDArray pred = manager.create(new float[0]);
            NDArray label = manager.create(new float[0]);
            // r = 2*(15 + 5) = 40
            L1WeightDecay decay =
                    Loss.l1WeightedDecay("", 2.0f, new NDList(parameters1, parameters2));
            Assert.assertEquals(
                    decay.evaluate(new NDList(label), new NDList(pred)).getFloat(), 40.0f);
        }
    }

    @Test
    public void l2DecayTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray parameters1 = manager.create(new float[] {-1, -2, 3, 4, 5}); // 55
            NDArray parameters2 = manager.create(new float[] {-1, -1, -1, -1, -1}); // 5
            // Not used
            NDArray pred = manager.create(new float[0]);
            NDArray label = manager.create(new float[0]);
            // r = 2*(55 + 5) = 120
            L2WeightDecay decay =
                    Loss.l2WeightedDecay("", 2.0f, new NDList(parameters1, parameters2));
            Assert.assertEquals(
                    decay.evaluate(new NDList(label), new NDList(pred)).getFloat(), 120.0f);
        }
    }

    @Test
    public void elasticNetDecayTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray parameters1 = manager.create(new float[] {-1, -2, 3, 4, 5});
            NDArray parameters2 = manager.create(new float[] {-1, -1, -1, -1, -1});
            // Not used
            NDArray pred = manager.create(new float[0]);
            NDArray label = manager.create(new float[0]);
            // r = L1 + L2 = 2*20 + 1*60 = 100
            ElasticNetWeightDecay decay =
                    Loss.elasticNetWeightedDecay(
                            "", 2.0f, 1.0f, new NDList(parameters1, parameters2));
            Assert.assertEquals(
                    decay.evaluate(new NDList(label), new NDList(pred)).getFloat(), 100.0f);
        }
    }
}
