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
package software.amazon.ai.integration.tests;

import java.util.Arrays;
import org.testng.annotations.Test;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.loss.Loss;

public class LossTest {

    @Test
    public void l1LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assertions.assertTrue(
                    Arrays.equals(
                            new float[] {0, 1, 2, 3, 4},
                            Loss.l1Loss().getLoss(label, pred).toFloatArray()));
        }
    }

    @Test
    public void l2LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assertions.assertTrue(
                    Arrays.equals(
                            new float[] {0f, 0.5f, 2f, 4.5f, 8f},
                            Loss.l2Loss().getLoss(label, pred).toFloatArray()));
        }
    }

    @Test
    public void softmaxCrossEntropyTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(1));
            Assertions.assertAlmostEquals(
                    manager.create(new float[] {3.45191431f}),
                    Loss.softmaxCrossEntropyLoss().getLoss(label, pred));
        }
    }

    @Test
    public void hingeLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5)).neg();
            Assertions.assertTrue(
                    Arrays.equals(
                            new float[] {2, 3, 4, 5, 6},
                            Loss.hingeLoss().getLoss(label, pred).toFloatArray()));
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
                    Loss.sigmoidBinaryCrossEntropyLoss().getLoss(label, pred));
        }
    }
}
