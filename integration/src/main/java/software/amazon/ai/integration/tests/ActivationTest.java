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
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.initializer.Initializer;

public class ActivationTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", ActivationTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testRelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDArray expected = manager.create(new float[] {0, 0, 2});
            Assertions.assertEquals(expected, Activation.relu(original));
            Assertions.assertEquals(new NDList(expected), Activation.relu(new NDList(original)));
            Assertions.assertEquals(expected, Activation.reluBlock().forward(original));
        }
    }

    @RunAsTest
    public void testSigmoid() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0.5f});
            Assertions.assertAlmostEquals(expected, Activation.sigmoid(original));
            Assertions.assertAlmostEquals(
                    new NDList(expected), Activation.sigmoid(new NDList(original)));
            Assertions.assertAlmostEquals(expected, Activation.sigmoidBlock().forward(original));
        }
    }

    @RunAsTest
    public void testTanh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.tanh(original));
            Assertions.assertEquals(new NDList(expected), Activation.tanh(new NDList(original)));
            Assertions.assertEquals(expected, Activation.tanhBlock().forward(original));
        }
    }

    @RunAsTest
    public void testSoftrelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0, 0, 2});
            NDArray expected = manager.create(new float[] {.6931f, .6931f, 2.1269f});
            Assertions.assertAlmostEquals(expected, Activation.softrelu(original));
            Assertions.assertAlmostEquals(
                    new NDList(expected), Activation.softrelu(new NDList(original)));
            Assertions.assertAlmostEquals(expected, Activation.softreluBlock().forward(original));
        }
    }

    @RunAsTest
    public void testLeakyrelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDArray expected = manager.create(new float[] {-1, 0, 2});
            float alpha = 1.0f;
            Assertions.assertEquals(expected, Activation.leakyRelu(original, alpha));
            Assertions.assertEquals(
                    new NDList(expected), Activation.leakyRelu(new NDList(original), alpha));
            Assertions.assertEquals(expected, Activation.leakyReluBlock(alpha).forward(original));
        }
    }

    @RunAsTest
    public void testElu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0, 2});
            NDArray expected = manager.create(new float[] {0, 2});
            float alpha = 1.0f;
            Assertions.assertEquals(expected, Activation.elu(original, alpha));
            Assertions.assertEquals(
                    new NDList(expected), Activation.elu(new NDList(original), alpha));
            Assertions.assertEquals(expected, Activation.eluBlock(alpha).forward(original));
        }
    }

    @RunAsTest
    public void testSelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.selu(original));
            Assertions.assertEquals(new NDList(expected), Activation.selu(new NDList(original)));
            Assertions.assertEquals(expected, Activation.seluBlock().forward(original));
        }
    }

    @RunAsTest
    public void testGelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            Assertions.assertEquals(expected, Activation.gelu(original));
            Assertions.assertEquals(new NDList(expected), Activation.gelu(new NDList(original)));
            Assertions.assertEquals(expected, Activation.geluBlock().forward(original));
        }
    }

    @RunAsTest
    public void testSwish() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {0});
            NDArray expected = manager.create(new float[] {0});
            float beta = 1.0f;
            Assertions.assertEquals(expected, Activation.swish(original, beta));
            Assertions.assertEquals(
                    new NDList(expected), Activation.swish(new NDList(original), beta));
            Assertions.assertEquals(expected, Activation.swishBlock(beta).forward(original));
        }
    }

    @RunAsTest
    public void testPrelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray original = manager.create(new float[] {-1, 0, 2});
            NDArray expected = manager.create(new float[] {-1, 0, 2});
            Activation block = Activation.preluBlock();
            block.setInitializer(manager, Initializer.ONES);
            Assertions.assertEquals(expected, block.forward(original));
        }
    }
}
