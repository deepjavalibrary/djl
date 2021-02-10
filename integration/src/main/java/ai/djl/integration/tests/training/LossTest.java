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
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
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
                    Loss.l1Loss().evaluate(new NDList(label), new NDList(pred)).getFloat(), 2.0f);
        }
    }

    @Test
    public void l2LossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assert.assertEquals(
                    Loss.l2Loss().evaluate(new NDList(label), new NDList(pred)).getFloat(), 3.0f);
        }
    }

    @Test
    public void softmaxCrossEntropyTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test fromLogits=true, sparseLabel=true
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(1));
            Assertions.assertAlmostEquals(
                    Loss.softmaxCrossEntropyLoss("loss", 1, -1, true, true)
                            .evaluate(new NDList(label), new NDList(pred)),
                    manager.create(3.45191431f));

            // test fromLogits=false, sparseLabel=true
            pred =
                    manager.create(
                            new float[] {4.0f, 2.0f, 1.0f, 0.0f, 5.0f, 1.0f}, new Shape(2, 3));
            label = manager.create(new float[] {0, 1}, new Shape(2));
            NDArray nonSparseLabel =
                    manager.create(new float[] {1f, 0f, 0f, 0f, 1f, 0f}, new Shape(2, 3));
            NDArray sparseOutput =
                    Loss.softmaxCrossEntropyLoss()
                            .evaluate(new NDList(label), new NDList(pred.logSoftmax(-1)));
            // test fromLogits=false, sparseLabel=false
            NDArray nonSparseOutput =
                    Loss.softmaxCrossEntropyLoss("loss", 1, -1, false, false)
                            .evaluate(new NDList(nonSparseLabel), new NDList(pred.logSoftmax(-1)));

            Assertions.assertAlmostEquals(sparseOutput, nonSparseOutput);
            Assertions.assertAlmostEquals(sparseOutput, manager.create(0.09729549f));
        }
    }

    @Test
    public void hingeLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5)).neg();
            Assert.assertEquals(
                    Loss.hingeLoss().evaluate(new NDList(label), new NDList(pred)).getFloat(),
                    4.0f);
        }
    }

    @Test
    public void sigmoidBinaryCrossEntropyLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.create(new float[] {1, 2, 3, 4, 5});
            NDArray label = manager.ones(new Shape(5));
            Assertions.assertAlmostEquals(
                    Loss.sigmoidBinaryCrossEntropyLoss()
                            .evaluate(new NDList(label), new NDList(pred))
                            .getFloat(),
                    0.10272846f);
        }
    }

    @Test
    public void maskedSoftmaxCrossEntropyLossTest() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray pred = manager.ones(new Shape(3, 4, 10));
            NDArray label = manager.ones(new Shape(3, 4));
            NDArray validLengths = manager.create(new int[] {4, 2, 0});
            Assertions.assertAlmostEquals(
                    Loss.maskedSoftmaxCrossEntropyLoss()
                            .evaluate(new NDList(label, validLengths), new NDList(pred)),
                    manager.create(new float[] {2.3025851f, 1.1512926f, 0}).reshape(3, 1));
        }
    }
}
