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
package ai.djl.mxnet.integration;

import ai.djl.Model;
import ai.djl.mxnet.engine.MxGradientCollector;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.testing.TestRequirements;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;

import org.testng.Assert;
import org.testng.annotations.Test;

public class MxGradientCollectorIntegrationTest {

    @Test
    public void testMxAutograd() {
        TestRequirements.notArm();

        MxGradientCollector.setRecording(false);
        MxGradientCollector.setTraining(false);
        try (Model model = Model.newInstance("model");
                NDManager manager = model.getNDManager()) {
            model.setBlock(Blocks.identityBlock());
            try (Trainer trainer =
                    model.newTrainer(
                            new DefaultTrainingConfig(Loss.l2Loss())
                                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT))) {
                try (GradientCollector gradCol = trainer.newGradientCollector()) {
                    NDArray lhs =
                            manager.create(new float[] {6, -9, -12, 15, 0, 4}, new Shape(2, 3));
                    NDArray rhs = manager.create(new float[] {2, 3, -4}, new Shape(3, 1));
                    NDArray expected =
                            manager.create(new float[] {2, 3, -4, 2, 3, -4}, new Shape(2, 3));
                    lhs.setRequiresGradient(true);
                    // autograd automatically set recording and training during initialization
                    Assert.assertTrue(MxGradientCollector.isRecording());
                    Assert.assertTrue(MxGradientCollector.isTraining());

                    NDArray result = NDArrays.dot(lhs, rhs);
                    gradCol.backward(result);
                    NDArray grad = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad, expected);
                    // test close and get again
                    grad.close();
                    NDArray grad2 = lhs.getGradient();
                    Assertions.assertAlmostEquals(grad2, expected);
                }
            }
        }
    }
}
