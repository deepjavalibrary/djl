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
import ai.djl.training.evaluator.AbstractAccuracy;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.evaluator.TopKAccuracy;
import org.testng.Assert;
import org.testng.annotations.Test;

public class EvaluatorTest {

    @Test
    public void testAccuracy() {
        try (NDManager manager = NDManager.newBaseManager()) {

            NDArray predictions =
                    manager.create(new float[] {0.3f, 0.7f, 0, 1, 0.4f, 0.6f}, new Shape(3, 2));
            NDArray labels = manager.create(new int[] {0, 1, 1}, new Shape(3));

            AbstractAccuracy acc = new Accuracy();
            acc.addAccumulator("");
            acc.updateAccumulator("", new NDList(labels), new NDList(predictions));
            float accuracy = acc.getAccumulator("");
            float expectedAccuracy = 2.f / 3;
            Assert.assertEquals(
                    accuracy,
                    expectedAccuracy,
                    "Wrong accuracy, expected: " + expectedAccuracy + ", actual: " + accuracy);

            labels = manager.create(new long[] {0, 1, 1}, new Shape(3));

            acc = new Accuracy();
            acc.addAccumulator("");
            acc.updateAccumulator("", new NDList(labels), new NDList(predictions));
            accuracy = acc.getAccumulator("");
            expectedAccuracy = 2.f / 3;
            Assert.assertEquals(
                    accuracy,
                    expectedAccuracy,
                    "Wrong accuracy, expected: " + expectedAccuracy + ", actual: " + accuracy);
        }
    }

    @Test
    public void testTopKAccuracy() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray predictions =
                    manager.create(
                            new float[] {
                                0.1f, 0.2f, 0.3f, 0.4f, 0, 1, 0, 0, 0.3f, 0.5f, 0.1f, 0.1f
                            },
                            new Shape(3, 4));
            NDArray labels = manager.create(new int[] {0, 1, 2}, new Shape(3));
            TopKAccuracy topKAccuracy = new TopKAccuracy(2);
            topKAccuracy.addAccumulator("");
            topKAccuracy.updateAccumulator("", new NDList(labels), new NDList(predictions));
            float expectedAccuracy = 1.f / 3;
            float accuracy = topKAccuracy.getAccumulator("");
            Assert.assertEquals(
                    accuracy,
                    expectedAccuracy,
                    "Wrong accuracy, expected: " + expectedAccuracy + ", actual: " + accuracy);

            labels = manager.create(new long[] {0, 1, 2}, new Shape(3));
            topKAccuracy = new TopKAccuracy(2);
            topKAccuracy.addAccumulator("");
            topKAccuracy.updateAccumulator("", new NDList(labels), new NDList(predictions));
            expectedAccuracy = 1.f / 3;
            accuracy = topKAccuracy.getAccumulator("");
            Assert.assertEquals(
                    accuracy,
                    expectedAccuracy,
                    "Wrong accuracy, expected: " + expectedAccuracy + ", actual: " + accuracy);
        }
    }
}
