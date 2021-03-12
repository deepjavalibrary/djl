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

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ActivationTest {

    TrainingConfig config =
            new DefaultTrainingConfig(Loss.l2Loss())
                    .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

    @Test
    public void testRelu() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.reluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(3));

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {-1, 0, 2});
                NDArray expected = manager.create(new float[] {0, 0, 2});
                Assert.assertEquals(Activation.relu(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testSigmoid() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.sigmoidBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0.5f});
                Assertions.assertAlmostEquals(Activation.sigmoid(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testTanh() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.tanhBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assert.assertEquals(Activation.tanh(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testSoftPlus() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.softPlusBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0, 0, 2});
                NDArray expected = manager.create(new float[] {.6931f, .6931f, 2.1269f});
                Assertions.assertAlmostEquals(Activation.softPlus(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testSoftSign() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.softSignBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0, 1, 2});
                NDArray expected = manager.create(new float[] {0, 0.5f, 0.6667f});
                Assertions.assertAlmostEquals(Activation.softSign(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testLeakyrelu() {
        try (Model model = Model.newInstance("model")) {
            float alpha = 1.0f;
            model.setBlock(Activation.leakyReluBlock(alpha));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {-1, 0, 2});
                NDArray expected = manager.create(new float[] {-1, 0, 2});
                Assert.assertEquals(Activation.leakyRelu(data, alpha), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testElu() {
        try (Model model = Model.newInstance("model")) {
            float alpha = 1.0f;
            model.setBlock(Activation.eluBlock(alpha));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0, 2});
                NDArray expected = manager.create(new float[] {0, 2});
                Assertions.assertAlmostEquals(Activation.elu(data, alpha), expected);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testSelu() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.seluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assert.assertEquals(Activation.selu(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testGelu() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.geluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assert.assertEquals(Activation.gelu(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }

    @Test
    public void testSwish() {
        try (Model model = Model.newInstance("model")) {
            float beta = 1.0f;
            model.setBlock(Activation.swishBlock(beta));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 5, 0.3f, 0.08f});
                NDArray expected = manager.create(new float[] {0.7311f, 4.9665f, 0.1723f, 0.0416f});
                Assertions.assertAlmostEquals(Activation.swish(data, beta), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testMish() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.mishBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 5, 0.3f, 0.08f});
                NDArray expected = manager.create(new float[] {0.9558f, 5, 0.253f, 0.0628f});
                Assertions.assertAlmostEquals(Activation.mish(data), expected);

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);
            }
        }
    }

    @Test
    public void testPrelu() {
        try (Model model = Model.newInstance("model")) {
            model.setBlock(Activation.preluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(3));
                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {-1, 0, 2});
                NDArray expected = manager.create(new float[] {-1, 0, 2});
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);
            }
        }
    }
}
