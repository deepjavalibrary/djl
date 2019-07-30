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

import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.training.metrics.Accuracy;

public class TrainingMetricsTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", TrainingMetricsTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testAccuracy() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            // test NDList
            NDList predictions = new NDList();
            predictions.add(manager.create(new float[] {0.3f, 0.7f}, new Shape(2, 1)));
            predictions.add(manager.create(new float[] {0, 1}, new Shape(2, 1)));
            predictions.add(manager.create(new float[] {0.4f, 0.6f}, new Shape(2, 1)));

            NDList labels = new NDList();
            labels.add(manager.create(new float[] {0}));
            labels.add(manager.create(new float[] {1}));
            labels.add(manager.create(new float[] {1}));

            NDArray predictionsNDArray =
                    manager.create(new float[] {0.3f, 0.7f, 0, 1, 0.4f, 0.6f}, new Shape(3, 2));
            NDArray labelsNDArray = manager.create(new float[] {0, 1, 1}, new Shape(3));

            Accuracy acc = new Accuracy();
            acc.update(labels, predictions);
            float accuracy = acc.getMetric().getValue().floatValue();
            float expectedAccuracy =
                    predictionsNDArray.argmax(1, false).eq(labelsNDArray).sum().getFloat()
                            / predictions.size();
            Assertions.assertEquals(
                    expectedAccuracy,
                    accuracy,
                    String.format(
                            "Wrong accuracy, expected: %f, actual: %f",
                            expectedAccuracy, accuracy));

            Accuracy accND = new Accuracy();
            accND.update(labelsNDArray, predictionsNDArray);
            float accuracy2 = accND.getMetric().getValue().floatValue();
            Assertions.assertEquals(
                    expectedAccuracy,
                    accuracy,
                    String.format(
                            "Wrong accuracy, expected: %f, actual: %f",
                            expectedAccuracy, accuracy2));
        }
    }
}
