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

import org.testng.annotations.Test;
import software.amazon.ai.Model;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.initializer.Initializer;

public class ActivationTest {

    private TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

    @Test
    public void testRelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.reluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {-1, 0, 2});
                NDArray expected = manager.create(new float[] {0, 0, 2});
                Assertions.assertEquals(expected, Activation.relu(original));
                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testSigmoid() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.sigmoidBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0.5f});
                Assertions.assertAlmostEquals(expected, Activation.sigmoid(original));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertAlmostEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testTanh() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.tanhBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assertions.assertEquals(expected, Activation.tanh(original));
                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testSoftrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.softreluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0, 0, 2});
                NDArray expected = manager.create(new float[] {.6931f, .6931f, 2.1269f});
                Assertions.assertAlmostEquals(expected, Activation.softrelu(original));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertAlmostEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testLeakyrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            float alpha = 1.0f;
            model.setBlock(Activation.leakyReluBlock(alpha));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {-1, 0, 2});
                NDArray expected = manager.create(new float[] {-1, 0, 2});
                Assertions.assertEquals(expected, Activation.leakyRelu(original, alpha));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertAlmostEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testElu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            float alpha = 1.0f;
            model.setBlock(Activation.eluBlock(alpha));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0, 2});
                NDArray expected = manager.create(new float[] {0, 2});
                Assertions.assertEquals(expected, Activation.elu(original, alpha));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testSelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.seluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assertions.assertEquals(expected, Activation.selu(original));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testGelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.geluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assertions.assertEquals(expected, Activation.gelu(original));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testSwish() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            float beta = 1.0f;
            model.setBlock(Activation.swishBlock(beta));

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray original = manager.create(new float[] {0});
                NDArray expected = manager.create(new float[] {0});
                Assertions.assertEquals(expected, Activation.swish(original, beta));

                NDList expectedList = new NDList(expected);
                NDList ndList = new NDList(original);
                Assertions.assertEquals(expectedList, trainer.forward(ndList));
            }
        }
    }

    @Test
    public void testPrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            model.setBlock(Activation.preluBlock());

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                NDArray original = manager.create(new float[] {-1, 0, 2});
                NDList expected = new NDList(manager.create(new float[] {-1, 0, 2}));
                Assertions.assertEquals(expected, trainer.forward(new NDList(original)));
            }
        }
    }
}
