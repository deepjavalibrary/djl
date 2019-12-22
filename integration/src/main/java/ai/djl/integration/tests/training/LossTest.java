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

import ai.djl.integration.util.Assertions;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LossTest {

    @Test
    public void l1LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertEquals(
                    Loss.l1Loss().getLoss(new NDList(label), new NDList(pred)).getFloat(), 2.0f);
        }
    }

    @Test
    public void l2LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertEquals(
                    Loss.l2Loss().getLoss(new NDList(label), new NDList(pred)).getFloat(), 3.0f);
        }
    }

    @Test
    public void softmaxCrossEntropyTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(1));
            Assertions.assertAlmostEquals(
                    Loss.softmaxCrossEntropyLoss().getLoss(new NDList(label), new NDList(pred)),
                    manager.create(3.45191431f));
        }
    }

    @Test
    public void hingeLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5)).neg();
            Assert.assertEquals(
                    Loss.hingeLoss().getLoss(new NDList(label), new NDList(pred)).getFloat(), 4.0f);
        }
    }

    @Test
    public void sigmoidBinaryCrossEntropyLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertEquals(
                    Loss.sigmoidBinaryCrossEntropyLoss()
                            .getLoss(new NDList(label), new NDList(pred))
                            .getFloat(),
                    0.10272846f);
        }
    }
}
