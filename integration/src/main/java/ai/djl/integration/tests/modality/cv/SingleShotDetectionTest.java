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

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.zoo.cv.object_detection.ssd.SingleShotDetection;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SingleShotDetectionTest {
    @Test
    public void testClassPredictorBlocks() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = SingleShotDetection.getClassPredictionBlock(5, 10);
            Assert.assertEquals(
                    block.getOutputShapes(manager, new Shape[] {new Shape(2, 8, 20, 20)})[0],
                    new Shape(2, 55, 20, 20));
            block = SingleShotDetection.getClassPredictionBlock(3, 10);
            Assert.assertEquals(
                    block.getOutputShapes(manager, new Shape[] {new Shape(2, 16, 10, 10)})[0],
                    new Shape(2, 33, 10, 10));
        }
    }

    @Test
    public void testAnchorPredictorBlocks() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block block = SingleShotDetection.getAnchorPredictionBlock(5);
            Assert.assertEquals(
                    block.getOutputShapes(manager, new Shape[] {new Shape(2, 8, 20, 20)})[0],
                    new Shape(2, 20, 20, 20));
            block = SingleShotDetection.getClassPredictionBlock(3, 10);
            Assert.assertEquals(
                    block.getOutputShapes(manager, new Shape[] {new Shape(2, 16, 10, 10)})[0],
                    new Shape(2, 33, 10, 10));
        }
    }

    @Test
    public void testDownSamplingBlock() {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block sequentialBlock = SingleShotDetection.getDownSamplingBlock(10);
            Assert.assertEquals(
                    sequentialBlock
                            .getOutputShapes(manager, new Shape[] {new Shape(2, 3, 20, 20)})[0],
                    new Shape(2, 10, 10, 10));
        }
    }
}
