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

import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.testing.Assertions;
import org.testng.annotations.Test;

public class MultiBoxDetectionTest {
    @Test
    public void testDetections() {
        MultiBoxDetection multiBoxDetection = MultiBoxDetection.builder().build();
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray anchors =
                    manager.create(
                                    new float[] {
                                        0.1f, 0.08f, 0.52f, 0.92f, 0.08f, 0.2f, 0.56f, 0.95f, 0.15f,
                                        0.3f, 0.62f, 0.91f, 0.55f, 0.2f, 0.9f, 0.88f
                                    })
                            .reshape(4, 4);
            NDArray offsetPreds = manager.zeros(new Shape(16));
            NDArray classProbs =
                    manager.create(
                                    new float[] {
                                        0, 0, 0, 0, 0.9f, 0.8f, 0.7f, 0.1f, 0.1f, 0.2f, 0.3f, 0.9f
                                    })
                            .reshape(3, 4);
            NDArray expected =
                    manager.create(
                                    new float[] {
                                        0, 0.9f, 0.1f, 0.08f, 0.52f, 0.92f, 1f, 0.9f, 0.55f, 0.2f,
                                        0.9f, 0.88f, -1f, 0.8f, 0.08f, 0.2f, 0.56f, 0.95f, -1f,
                                        0.7f, 0.15f, 0.3f, 0.62f, 0.91f
                                    })
                            .reshape(1, 4, 6);
            NDArray actual =
                    multiBoxDetection
                            .detection(
                                    new NDList(
                                            classProbs.expandDims(0),
                                            offsetPreds.expandDims(0),
                                            anchors.expandDims(0)))
                            .head();
            // orders of detection results is not the same on CPU and GPU
            // but does not affect detection correctness
            Assertions.assertAlmostEquals(actual.sort(1), expected.sort(1));
        }
    }
}
