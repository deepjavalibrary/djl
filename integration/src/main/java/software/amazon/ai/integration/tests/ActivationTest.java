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

import java.util.Arrays;
import java.util.stream.Stream;
import software.amazon.ai.Model;
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.initializer.Initializer;

public class ActivationTest {

    public static void main(String[] args) {
        String[] cmd = {"-c", ActivationTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testRelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDArray expected = manager.create(new float[] {0, 0, 2});
            Assertions.assertEquals(expected, Activation.relu(original));
            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.relu(ndList));
            Assertions.assertEquals(expectedList, Activation.reluBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testSigmoid() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0.5f});
            Assertions.assertAlmostEquals(expected, Activation.sigmoid(original));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertAlmostEquals(expectedList, Activation.sigmoid(ndList));
            Assertions.assertAlmostEquals(expectedList, Activation.sigmoidBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testTanh() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.tanh(original));
            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.tanh(ndList));
            Assertions.assertEquals(expectedList, Activation.tanhBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testSoftrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0, 0, 2});
            NDArray expected = manager.create(new float[] {.6931f, .6931f, 2.1269f});
            Assertions.assertAlmostEquals(expected, Activation.softrelu(original));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertAlmostEquals(expectedList, Activation.softrelu(ndList));
            Assertions.assertAlmostEquals(expectedList, Activation.softreluBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testLeakyrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDArray expected = manager.create(new float[] {-1, 0, 2});
            float alpha = 1.0f;
            Assertions.assertEquals(expected, Activation.leakyRelu(original, alpha));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertAlmostEquals(expectedList, Activation.leakyRelu(ndList, alpha));
            Assertions.assertAlmostEquals(
                    expectedList, Activation.leakyReluBlock(alpha).forward(ndList));
        }
    }

    @RunAsTest
    public void testElu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0, 2});
            NDArray expected = manager.create(new float[] {0, 2});
            float alpha = 1.0f;
            Assertions.assertEquals(expected, Activation.elu(original, alpha));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.elu(ndList, alpha));
            Assertions.assertEquals(expectedList, Activation.eluBlock(alpha).forward(ndList));
        }
    }

    @RunAsTest
    public void testSelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.selu(original));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.selu(ndList));
            Assertions.assertEquals(expectedList, Activation.seluBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testGelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.gelu(original));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.gelu(ndList));
            Assertions.assertEquals(expectedList, Activation.geluBlock().forward(ndList));
        }
    }

    @RunAsTest
    public void testSwish() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            float beta = 1.0f;
            Assertions.assertEquals(expected, Activation.swish(original, beta));

            NDList expectedList = new NDList(expected);
            NDList ndList = new NDList(original);
            Assertions.assertEquals(expectedList, Activation.swish(ndList, beta));
            Assertions.assertEquals(expectedList, Activation.swishBlock(beta).forward(ndList));
        }
    }

    @RunAsTest
    public void testPrelu() throws FailedTestException {
        try (Model model = Model.newInstance()) {
            NDManager manager = model.getNDManager();
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDList expected = new NDList(manager.create(new float[] {-1, 0, 2}));
            Block block = Activation.preluBlock();
            block.setInitializer(manager, Initializer.ONES);
            Assertions.assertEquals(expected, block.forward(new NDList(original)));
        }
    }
}
