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

import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.ParameterStore;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SingleShotDetectionTest {

    @Test
    public void testClassPredictorBlocks() {
        Block block = SingleShotDetection.getClassPredictionBlock(5, 10);
        Assert.assertEquals(
                block.getOutputShapes(new Shape[] {new Shape(2, 8, 20, 20)})[0],
                new Shape(2, 55, 20, 20));
        block = SingleShotDetection.getClassPredictionBlock(3, 10);
        Assert.assertEquals(
                block.getOutputShapes(new Shape[] {new Shape(2, 16, 10, 10)})[0],
                new Shape(2, 33, 10, 10));
    }

    @Test
    public void testAnchorPredictorBlocks() {
        Block block = SingleShotDetection.getAnchorPredictionBlock(5);
        Assert.assertEquals(
                block.getOutputShapes(new Shape[] {new Shape(2, 8, 20, 20)})[0],
                new Shape(2, 20, 20, 20));
        block = SingleShotDetection.getClassPredictionBlock(3, 10);
        Assert.assertEquals(
                block.getOutputShapes(new Shape[] {new Shape(2, 16, 10, 10)})[0],
                new Shape(2, 33, 10, 10));
    }

    @Test
    public void testDownSamplingBlock() {
        Block sequentialBlock = SingleShotDetection.getDownSamplingBlock(10);
        Assert.assertEquals(
                sequentialBlock.getOutputShapes(new Shape[] {new Shape(2, 3, 20, 20)})[0],
                new Shape(2, 10, 10, 10));
    }

    @Test
    public void testSingleShotDetectionShape() {
        try (NDManager manager = NDManager.newBaseManager()) {
            int[] numFilters = {16, 32, 64};
            SequentialBlock block = new SequentialBlock();
            for (int numFilter : numFilters) {
                block.add(SingleShotDetection.getDownSamplingBlock(numFilter));
            }

            List<List<Float>> sizes = new ArrayList<>();
            List<List<Float>> ratios = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                ratios.add(Arrays.asList(1f, 2f, 0.5f));
            }
            sizes.add(Arrays.asList(0.2f, 0.272f));
            sizes.add(Arrays.asList(0.37f, 0.447f));
            sizes.add(Arrays.asList(0.54f, 0.619f));
            sizes.add(Arrays.asList(0.71f, 0.79f));
            sizes.add(Arrays.asList(0.88f, 0.961f));

            SingleShotDetection ssd =
                    SingleShotDetection.builder()
                            .setNumClasses(1)
                            .setNumFeatures(3)
                            .optGlobalPool(true)
                            .setRatios(ratios)
                            .setSizes(sizes)
                            .setBaseNetwork(block)
                            .build();
            ssd.initialize(manager, DataType.FLOAT32, new Shape(32, 3, 256, 256));
            ParameterStore ps = new ParameterStore(manager, false);
            NDList output =
                    ssd.forward(ps, new NDList(manager.ones(new Shape(32, 3, 256, 256))), false);
            Assert.assertEquals(output.get(0).getShape(), new Shape(1, 5444, 4));
            Assert.assertEquals(output.get(1).getShape(), new Shape(32, 5444, 2));
            Assert.assertEquals(output.get(2).getShape(), new Shape(32, 21776));
            Shape[] outputShapes = ssd.getOutputShapes(new Shape[] {new Shape(32, 3, 256, 256)});
            Assert.assertEquals(outputShapes[0], new Shape(1, 5444, 4));
            Assert.assertEquals(outputShapes[1], new Shape(32, 5444, 2));
            Assert.assertEquals(outputShapes[2], new Shape(32, 21776));
        }
    }
}
