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
import software.amazon.ai.integration.IntegrationTest;
import software.amazon.ai.integration.exceptions.FailedTestException;
import software.amazon.ai.integration.util.Assertions;
import software.amazon.ai.integration.util.RunAsTest;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.convolutional.Conv3D;
import software.amazon.ai.nn.core.Embedding;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.norm.BatchNorm;
import software.amazon.ai.nn.norm.Dropout;
import software.amazon.ai.nn.recurrent.GRU;
import software.amazon.ai.nn.recurrent.LSTM;
import software.amazon.ai.nn.recurrent.RNN;
import software.amazon.ai.training.initializer.Initializer;

public class BlockCoreTest {

    public static void main(String[] args) {
        String[] cmd = new String[] {"-c", BlockCoreTest.class.getName()};
        new IntegrationTest().runTests(cmd);
    }

    @RunAsTest
    public void testLinear() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            long outSize = 3;

            Linear linearWithBias = new Linear.Builder().setOutChannels(outSize).build();
            linearWithBias.setInitializer(manager, Initializer.ONES);
            NDArray outBias = linearWithBias.forward(input);
            NDArray expectedBias =
                    input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                            .add(manager.ones(new Shape(2, outSize)));
            Assertions.assertEquals(expectedBias, outBias);

            Linear linearWithoutBias =
                    new Linear.Builder().setOutChannels(outSize).setBias(false).build();
            linearWithoutBias.setInitializer(manager, Initializer.ONES);
            NDArray outNoBias = linearWithoutBias.forward(input);
            NDArray expectedNoBias = input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
            Assertions.assertEquals(expectedNoBias, outNoBias);
        }
    }

    @RunAsTest
    public void testLinearWithDefinedLayout() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {1, 2, 3, 4},
                            new Shape(
                                    new long[] {2, 2},
                                    new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL}));
            long outSize = 3;

            Linear linearWithBias = new Linear.Builder().setOutChannels(outSize).build();
            linearWithBias.setInitializer(manager, Initializer.ONES);
            NDArray outBias = linearWithBias.forward(input);
            NDArray expectedBias =
                    input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                            .add(manager.ones(new Shape(2, outSize)));
            Assertions.assertEquals(expectedBias, outBias);

            Linear linearWithoutBias =
                    new Linear.Builder().setOutChannels(outSize).setBias(false).build();
            linearWithoutBias.setInitializer(manager, Initializer.ONES);
            NDArray outNoBias = linearWithoutBias.forward(input);
            NDArray expectedNoBias = input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
            Assertions.assertEquals(expectedNoBias, outNoBias);
        }
    }

    @RunAsTest
    public void testBatchNorm() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            NDArray expected = manager.create(new float[] {0, 1, 2, 3}, new Shape(2, 2));
            BatchNorm bn = new BatchNorm.Builder().setAxis(1).build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertAlmostEquals(expected, out);
        }
    }

    @RunAsTest
    public void testDropout() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
            Dropout dropout = new Dropout.Builder().setProbability(.5f).build();
            NDArray out = dropout.forward(input);
            Assertions.assertTrue(out.lte(out).all());
        }
    }

    @RunAsTest
    public void testEmbedding() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Embedding<Character> block =
                    new Embedding.Builder<Character>()
                            .setItems(Arrays.asList('a', 'b', 'c'))
                            .setEmbeddingSize(2)
                            .build();
            block.setInitializer(manager, Initializer.ONES);
            Assertions.assertEquals(manager.create(new int[] {1, 1}), block.forward(manager, 'x'));
            Assertions.assertEquals(
                    manager.create(new int[] {1, 1, 1, 1}, new Shape(2, 2)),
                    block.forward(manager, new Character[] {'a', 'b'}));
        }
    }

    @RunAsTest
    public void testConv1D() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 4, 4));
            NDArray expected = manager.create(new float[] {62, 56, 45}, new Shape(1, 1, 3));
            Conv1D bn = new Conv1D.Builder().setKernel(new Shape(2)).setNumFilters(1).build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertEquals(expected, out);
            Assertions.assertTrue(out.getShape().equals(bn.getOutputShape(new Shape(1, 4, 4))));
        }
    }

    @RunAsTest
    public void testConv2D() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                            new Shape(1, 1, 4, 4));
            NDArray expected =
                    manager.create(
                            new float[] {23, 25, 26, 22, 27, 24, 40, 32, 20},
                            new Shape(1, 1, 3, 3));
            Conv2D bn = new Conv2D.Builder().setKernel(new Shape(2, 2)).setNumFilters(1).build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertAlmostEquals(expected, out);
        }
    }

    @RunAsTest
    public void testConv3D() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input =
                    manager.create(
                            new float[] {
                                9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4, 4, 9, 7, 5, 11,
                                2, 5, 13, 10, 8, 4
                            },
                            new Shape(1, 1, 3, 3, 3));
            NDArray expected =
                    manager.create(
                            new float[] {61, 42, 55, 49, 56, 60, 57, 62}, new Shape(1, 1, 2, 2, 2));
            Conv3D bn = new Conv3D.Builder().setKernel(new Shape(2, 2, 2)).setNumFilters(1).build();
            bn.setInitializer(manager, Initializer.ONES);
            NDArray out = bn.forward(input);
            Assertions.assertEquals(expected, out);
            Assertions.assertTrue(
                    out.getShape().equals(bn.getOutputShape(new Shape(1, 1, 3, 3, 3))));
        }
    }

    @RunAsTest
    public void testRNNTanh() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.arange(0, 48, 1).reshape(new Shape(3, 4, 4));
            RNN rnn =
                    new RNN.Builder()
                            .setStateSize(5)
                            .setNumStackedLayers(1)
                            .setActivation(RNN.Activation.TANH)
                            .build();
            rnn.setInitializer(manager, Initializer.ONES);
            NDList outputs = rnn.forward(new NDList(input));
            NDArray out = outputs.get(0);
            Assertions.assertEquals(manager.ones(new Shape(3, 4, 5)), out);
        }
    }

    @RunAsTest
    public void testRNNRelu() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));
            RNN rnn =
                    new RNN.Builder()
                            .setStateSize(5)
                            .setNumStackedLayers(1)
                            .setActivation(RNN.Activation.RELU)
                            .build();
            rnn.setInitializer(manager, Initializer.ONES);
            NDList outputs = rnn.forward(new NDList(input));
            NDArray out = outputs.get(0);
            NDArray expected =
                    manager.create(
                            new float[] {13, 13, 13, 13, 13, 29, 29, 29, 29, 29},
                            new Shape(1, 2, 5));
            Assertions.assertEquals(expected, out);
        }
    }

    @RunAsTest
    public void testLSTM() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));
            LSTM lstm =
                    new LSTM.Builder()
                            .setStateSize(4)
                            .setNumStackedLayers(1)
                            .setActivation(RNN.Activation.RELU)
                            .build();
            lstm.setInitializer(manager, Initializer.ONES);
            NDList outputs = lstm.forward(new NDList(input));
            NDArray out = outputs.get(0);
            NDArray expected =
                    manager.create(
                            new float[] {
                                0.9640276f,
                                0.9640276f,
                                0.9640276f,
                                0.9640276f,
                                0.9640276f,
                                0.9640276f,
                                0.9640276f,
                                0.9640276f
                            },
                            new Shape(1, 2, 4));
            Assertions.assertAlmostEquals(expected, out);
        }
    }

    @RunAsTest
    public void testGRU() throws FailedTestException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));
            GRU lstm =
                    new GRU.Builder()
                            .setStateSize(4)
                            .setNumStackedLayers(1)
                            .setActivation(RNN.Activation.RELU)
                            .build();
            lstm.setInitializer(manager, Initializer.ONES);
            NDList outputs = lstm.forward(new NDList(input));
            NDArray out = outputs.get(0);
            Assertions.assertAlmostEquals(manager.ones(new Shape(1, 2, 4)), out);
        }
    }
}
