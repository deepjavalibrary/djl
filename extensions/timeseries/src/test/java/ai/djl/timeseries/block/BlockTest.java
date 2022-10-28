/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;
import java.util.List;

public class BlockTest {

    @Test
    public void testFeature() {
        int n = 10;
        int t = 20;

        // single static feat
        Shape inputShape = new Shape(n, 1);
        List<Integer> cardinalities = Arrays.asList(50);
        List<Integer> embeddingDims = Arrays.asList(10);
        testFeatureEmbedder(inputShape, cardinalities, embeddingDims);

        // single dynamic feat
        inputShape = new Shape(n, t, 1);
        cardinalities = Arrays.asList(2);
        testFeatureEmbedder(inputShape, cardinalities, embeddingDims);

        // multiple static feat
        inputShape = new Shape(n, 4);
        cardinalities = Arrays.asList(50, 50, 50, 50);
        embeddingDims = Arrays.asList(10, 20, 30, 40);
        testFeatureEmbedder(inputShape, cardinalities, embeddingDims);

        // multiple dynamic features
        inputShape = new Shape(n, t, 3);
        cardinalities = Arrays.asList(30, 30, 30);
        embeddingDims = Arrays.asList(10, 20, 30);
        testFeatureEmbedder(inputShape, cardinalities, embeddingDims);
    }

    @Test
    public void testScaler() {
        try (NDManager manager = NDManager.newBaseManager()) {
            ParameterStore ps = new ParameterStore(manager, false);

            Scaler scaler = MeanScaler.builder().setDim(1).build();
            NDArray target = manager.randomNormal(new Shape(5, 30));
            NDArray observed = manager.zeros(new Shape(5, 30));
            NDArray expScale = manager.full(new Shape(5), 1e-10f);
            Assert.assertEquals(
                    scaler.getOutputShapes(new Shape[] {target.getShape(), observed.getShape()})[1],
                    expScale.getShape());
            assertOutput(
                    scaler.forward(ps, new NDList(target, observed), false),
                    target,
                    expScale,
                    1,
                    false);

            scaler = MeanScaler.builder().setDim(2).optMinimumScale(1e-6f).build();
            target = manager.randomNormal(new Shape(5, 3, 30));
            observed = manager.zeros(new Shape(5, 3, 30));
            expScale = manager.full(new Shape(5, 3), 1e-6f);
            Assert.assertEquals(
                    scaler.getOutputShapes(new Shape[] {target.getShape(), observed.getShape()})[1],
                    expScale.getShape());
            assertOutput(
                    scaler.forward(ps, new NDList(target, observed), false),
                    target,
                    expScale,
                    2,
                    false);

            scaler = MeanScaler.builder().setDim(1).optKeepDim(true).build();
            target = manager.randomNormal(new Shape(5, 30, 1));
            observed = manager.zeros(new Shape(5, 30, 1));
            expScale = manager.full(new Shape(5, 1, 1), 1e-10f);
            Assert.assertEquals(
                    scaler.getOutputShapes(new Shape[] {target.getShape(), observed.getShape()})[1],
                    expScale.getShape());
            assertOutput(
                    scaler.forward(ps, new NDList(target, observed), false),
                    target,
                    expScale,
                    1,
                    true);

            scaler = NopScaler.builder().setDim(1).build();
            target = manager.randomNormal(new Shape(10, 20, 30));
            observed = manager.zeros(new Shape(10, 20, 30));
            expScale = manager.ones(new Shape(10, 30));
            Assert.assertEquals(
                    scaler.getOutputShapes(new Shape[] {target.getShape(), observed.getShape()})[1],
                    expScale.getShape());
            assertOutput(
                    scaler.forward(ps, new NDList(target, observed), false),
                    target,
                    expScale,
                    1,
                    false);

            scaler = NopScaler.builder().setDim(1).optKeepDim(true).build();
            target = manager.randomNormal(new Shape(10, 20, 30));
            observed = manager.ones(new Shape(10, 20, 30));
            expScale = manager.ones(new Shape(10, 1, 30));
            Assert.assertEquals(
                    scaler.getOutputShapes(new Shape[] {target.getShape(), observed.getShape()})[1],
                    expScale.getShape());
            assertOutput(
                    scaler.forward(ps, new NDList(target, observed), false),
                    target,
                    expScale,
                    1,
                    true);
        }
    }

    private void assertOutput(
            NDList scalerOutput, NDArray target, NDArray expScale, int dim, boolean keepDim) {
        NDArray actTargetScaled = scalerOutput.get(0);
        NDArray actScale = scalerOutput.get(1);
        Assertions.assertAlmostEquals(actScale, expScale);

        NDArray expTargetScaled;
        if (keepDim) {
            expTargetScaled = target.div(expScale);
        } else {
            expTargetScaled = target.div(expScale.expandDims(dim));
        }
        Assertions.assertAlmostEquals(actTargetScaled, expTargetScaled);
    }

    private void testFeatureEmbedder(
            Shape inputShape, List<Integer> cardinalities, List<Integer> embeddingDims) {
        try (NDManager manager = NDManager.newBaseManager()) {
            Shape outputShape =
                    inputShape
                            .slice(0, inputShape.dimension() - 1)
                            .add(embeddingDims.stream().mapToInt(Integer::intValue).sum());

            FeatureEmbedder embedder =
                    FeatureEmbedder.builder()
                            .setCardinalities(cardinalities)
                            .setEmbeddingDims(embeddingDims)
                            .build();

            embedder.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            embedder.initialize(manager, DataType.FLOAT32, inputShape);

            int expParamsLen = embedder.getParameters().keys().size();
            int actParamsLen = embeddingDims.size();
            Assert.assertEquals(expParamsLen, actParamsLen);

            ParameterStore ps = new ParameterStore(manager, true);
            NDArray actOutput =
                    embedder.forward(ps, new NDList(manager.ones(inputShape)), true)
                            .singletonOrThrow();
            NDArray expOutput = manager.ones(outputShape);
            Assert.assertEquals(actOutput.getShape(), expOutput.getShape());
            Assertions.assertAlmostEquals(actOutput, expOutput);
        }
    }
}
