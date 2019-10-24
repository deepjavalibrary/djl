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
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class LossTest {

    @Test
    public void l1LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertTrue(
                    Arrays.equals(
                            new float[] {0, 1, 2, 3, 4},
                            Loss.l1Loss()
                                    .getLoss(new NDList(label), new NDList(pred))
                                    .toFloatArray()));
        }
    }

    @Test
    public void l2LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertTrue(
                    Arrays.equals(
                            new float[] {0f, 0.5f, 2f, 4.5f, 8f},
                            Loss.l2Loss()
                                    .getLoss(new NDList(label), new NDList(pred))
                                    .toFloatArray()));
        }
    }

    @Test
    public void softmaxCrossEntropyTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(1));
            Assertions.assertAlmostEquals(
                    manager.create(new float[] {3.45191431f}),
                    Loss.softmaxCrossEntropyLoss().getLoss(new NDList(label), new NDList(pred)));
        }
    }

    @Test
    public void hingeLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5)).neg();
            Assert.assertTrue(
                    Arrays.equals(
                            new float[] {2, 3, 4, 5, 6},
                            Loss.hingeLoss()
                                    .getLoss(new NDList(label), new NDList(pred))
                                    .toFloatArray()));
        }
    }

    @Test
    public void sigmoidBinaryCrossEntropyLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assertions.assertAlmostEquals(
                    manager.create(
                            new float[] {
                                0.31326175f, 0.12692809f, 0.04858732f, 0.01814985f, 0.0067153f
                            }),
                    Loss.sigmoidBinaryCrossEntropyLoss()
                            .getLoss(new NDList(label), new NDList(pred)));
        }
    }
}
