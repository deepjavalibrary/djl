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
package ai.djl.integration.tests.modality.cv;

import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import org.testng.Assert;
import org.testng.annotations.Test;

public class MultiBoxTargetTest {
    @Test
    public void testTargets() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray anchorBoxes = manager.arange(22840.0f * 4.0f).reshape(new Shape(1, 22840, 4));
            NDArray label = manager.arange(160.0f).reshape(new Shape(32, 1, 5));
            NDArray classPreds =
                    manager.arange(32.0f * 22840.0f * 2.0f).reshape(new Shape(32, 2, 22840));

            MultiBoxTarget multiBoxTarget = MultiBoxTarget.builder().build();
            NDList targets = multiBoxTarget.target(new NDList(anchorBoxes, label, classPreds));

            Assert.assertEquals(targets.get(0).getShape(), new Shape(32, 91360));
            Assert.assertEquals(targets.get(1).getShape(), new Shape(32, 91360));
            Assert.assertEquals(targets.get(2).getShape(), new Shape(32, 22840));
        }
    }

    @Test
    public void testMultiBoxTargetValues() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray groundTruth =
                    manager.create(
                                    new float[] {
                                        0f, 0.1f, 0.08f, 0.52f, 0.92f, 1f, 0.55f, 0.2f, 0.9f, 0.88f
                                    })
                            .reshape(new Shape(1, 2, 5));
            NDArray anchorBoxes =
                    manager.create(
                                    new float[] {
                                        0f, 0.1f, 0.2f, 0.3f, 0.15f, 0.2f, 0.4f, 0.4f, 0.63f, 0.05f,
                                        0.88f, 0.98f, 0.66f, 0.45f, 0.8f, 0.8f, 0.57f, 0.3f, 0.92f,
                                        0.9f
                                    })
                            .reshape(1, 5, 4);

            MultiBoxTarget multiBoxTarget = MultiBoxTarget.builder().build();
            NDList targets =
                    multiBoxTarget.target(
                            new NDList(
                                    anchorBoxes, groundTruth, manager.zeros(new Shape(1, 3, 5))));

            Assertions.assertAlmostEquals(
                    targets.get(2),
                    manager.create(new float[] {0, 1, 2, 0, 2}).reshape(new Shape(1, 5)));
            Assertions.assertAlmostEquals(
                    targets.get(1),
                    manager.create(
                                    new float[] {
                                        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1
                                    })
                            .reshape(new Shape(1, 20)));
            Assertions.assertAlmostEquals(
                    targets.get(0),
                    manager.create(
                                    new float[] {
                                        0.00e+00f,
                                        0.00e+00f,
                                        0.00e+00f,
                                        0.00e+00f,
                                        1.40e+00f,
                                        1.00e+01f,
                                        2.593e+00f,
                                        7.175e+00f,
                                        -1.20e+00f,
                                        2.688e-01f,
                                        1.6823e+00f,
                                        -1.5654e+00f,
                                        0.00e+00f,
                                        0.00e+00f,
                                        0.00e+00f,
                                        0.00e+00f,
                                        -5.714e-01f,
                                        -1.00e+00f,
                                        -8.94e-07f,
                                        6.258e-01f
                                    })
                            .reshape(new Shape(1, 20)));
        }
    }
}
