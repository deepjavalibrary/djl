/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.integration.tests;

import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.integration.util.TestUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.patch.BasicParamPatch;
import ai.djl.patch.ParamPatch;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Tests for {@link ai.djl.patch.Patch}. */
public class PatchTest {

    @Test
    public void testScaleReverse() {
        try (Model model = Model.newInstance("model", TestUtils.getEngine())) {
            testMlp(model);
            double initialParamSum = paramSum(model);

            // Create patch
            Map<String, NDArray> patchData = new ConcurrentHashMap<>();
            for (Pair<String, Parameter> param : model.getBlock().getParameters()) {
                patchData.put(param.getKey(), param.getValue().getArray().onesLike());
            }
            try (BasicParamPatch patch = new BasicParamPatch(patchData)) {
                patch.scale(3).apply(model);
                Assert.assertEquals(paramSum(model), initialParamSum + 3 * paramSize(model));

                patch.reverse().apply(model);
                Assert.assertEquals(paramSum(model), initialParamSum + 2 * paramSize(model));

                patch.scale(-2).apply(model);
                Assert.assertEquals(paramSum(model), initialParamSum);
            }
        }
    }

    @Test
    public void testComparison() {
        try (Model model0 = Model.newInstance("m0", TestUtils.getEngine());
                Model model1 = Model.newInstance("m1", TestUtils.getEngine())) {
            testMlp(model0, Initializer.ZEROS);
            testMlp(model1, Initializer.ONES);

            ParamPatch patch = BasicParamPatch.makePatch(model0, model1);
            patch.apply(model0);
            Assert.assertEquals(paramSum(model1), paramSum(model0));
        }
    }

    @Test
    public void testGradients() {
        try (Model model = Model.newInstance("model", TestUtils.getEngine())) {
            testMlp(model, null);
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optOptimizer(
                                            Optimizer.sgd()
                                                    .setLearningRateTracker(Tracker.fixed(1.0f))
                                                    .build())
                                    .optInitializer(Initializer.ONES, p -> true))) {
                trainer.initialize(new Shape(10));
                double initialParamSum = paramSum(model);
                ParamPatch patch;
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDList preds =
                            trainer.forward(
                                    new NDList(model.getNDManager().ones(new Shape(10))),
                                    new NDList(model.getNDManager().ones(new Shape(10))));
                    NDArray loss =
                            trainer.getLoss()
                                    .evaluate(
                                            new NDList(
                                                    model.getNDManager().full(new Shape(1), 100)),
                                            preds);
                    collector.backward(loss);
                    patch = BasicParamPatch.makePatch(trainer.getModel(), collector);
                }
                trainer.step();

                Assert.assertNotEquals(paramSum(model), initialParamSum);
                // Note that to reverse a gradient update, you must also account for learning rate
                patch.reverse().apply(model);
                Assert.assertEquals(paramSum(model), initialParamSum);
            }
        }
    }

    private double paramSum(Model model) {
        return model.getBlock().getParameters().values().stream()
                .mapToDouble(p -> p.getArray().sum().toType(DataType.FLOAT32, true).getFloat())
                .sum();
    }

    private long paramSize(Model model) {
        return model.getBlock().getParameters().values().stream()
                .mapToLong(p -> p.getArray().getShape().size())
                .sum();
    }

    private void testMlp(Model model) {
        testMlp(model, Initializer.ONES);
    }

    private void testMlp(Model model, Initializer initializer) {
        Mlp block = new Mlp(10, 1, new int[] {10});
        if (initializer != null) {
            block.setInitializer(initializer, p -> true);
            block.initialize(model.getNDManager(), DataType.FLOAT32, new Shape(10));
        }
        model.setBlock(block);
    }
}
