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

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
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
import software.amazon.ai.ndarray.types.LayoutType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.Parameter;
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
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.util.PairList;

public class BlockCoreTest {

    public static void main(String[] args) {
        String[] cmd = {"-c", BlockCoreTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    @RunAsTest
    public void testLinear() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        long outSize = 3;
        Block block = new Linear.Builder().setOutChannels(outSize).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));

                NDArray outBias = trainer.forward(new NDList(input)).head();
                NDArray expectedBias =
                        input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                                .add(manager.ones(new Shape(2, outSize)));
                Assertions.assertEquals(expectedBias, outBias);

                testEncode(manager, block);
            }
        }

        block = new Linear.Builder().setOutChannels(outSize).setBias(false).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));

                NDArray outNoBias = trainer.forward(new NDList(input)).head();
                NDArray expectedNoBias =
                        input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
                Assertions.assertEquals(expectedNoBias, outNoBias);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testLinearWithDefinedLayout() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        long outSize = 3;
        Block block = new Linear.Builder().setOutChannels(outSize).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input =
                        manager.create(
                                new float[] {1, 2, 3, 4},
                                new Shape(
                                        new long[] {2, 2},
                                        new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL}));

                NDArray outBias = trainer.forward(new NDList(input)).head();
                NDArray expectedBias =
                        input.mmul(manager.ones(new Shape(outSize, 2)).transpose())
                                .add(manager.ones(new Shape(2, outSize)));
                Assertions.assertEquals(expectedBias, outBias);

                testEncode(manager, block);
            }
        }

        block = new Linear.Builder().setOutChannels(outSize).setBias(false).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input =
                        manager.create(
                                new float[] {1, 2, 3, 4},
                                new Shape(
                                        new long[] {2, 2},
                                        new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL}));

                NDArray outNoBias = trainer.forward(new NDList(input)).head();
                NDArray expectedNoBias =
                        input.mmul(manager.ones(new Shape(outSize, 2)).transpose());
                Assertions.assertEquals(expectedNoBias, outNoBias);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testBatchNorm() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        Block block = new BatchNorm.Builder().setAxis(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
                NDArray expected = manager.create(new float[] {0, 1, 2, 3}, new Shape(2, 2));

                NDArray out = trainer.forward(new NDList(input)).head();
                Assertions.assertAlmostEquals(expected, out);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testDropout() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        Block block = new Dropout.Builder().setProbability(.5f).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                NDArray input = manager.create(new float[] {1, 2, 3, 4}, new Shape(2, 2));
                NDArray out = trainer.forward(new NDList(input)).head();
                Assertions.assertTrue(out.lte(out).all());

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testEmbedding() throws FailedTestException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        Embedding<Character> block =
                new Embedding.Builder<Character>()
                        .setItems(Arrays.asList('a', 'b', 'c'))
                        .setEmbeddingSize(2)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                // TODO: use trainer.forward
                Assertions.assertEquals(
                        manager.create(new int[] {1, 1}), block.forward(manager, 'x'));
                Assertions.assertEquals(
                        manager.create(new int[] {1, 1, 1, 1}, new Shape(2, 2)),
                        block.forward(manager, new Character[] {'a', 'b'}));
            }
        }
    }

    @RunAsTest
    public void testConv1D() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        Block block =
                new Conv1D.Builder()
                        .setKernel(new Shape(2))
                        .setNumFilters(1)
                        .setBias(false)
                        .build();

        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input =
                        manager.create(
                                new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                                new Shape(1, 4, 4));
                NDArray expected = manager.create(new float[] {61, 55, 44}, new Shape(1, 1, 3));
                NDArray out = trainer.forward(new NDList(input)).get(0);
                Assertions.assertEquals(expected, out);

                Shape outputShape = block.getOutputShape(new Shape(1, 4, 4));
                Assertions.assertTrue(out.getShape().equals(outputShape));

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testConv2D() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        Block block = new Conv2D.Builder().setKernel(new Shape(2, 2)).setNumFilters(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input =
                        manager.create(
                                new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                                new Shape(1, 1, 4, 4));
                NDArray expected =
                        manager.create(
                                new float[] {23, 25, 26, 22, 27, 24, 40, 32, 20},
                                new Shape(1, 1, 3, 3));

                NDArray out = trainer.forward(new NDList(input)).get(0);
                Assertions.assertAlmostEquals(expected, out);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testConv3D() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        Block block = new Conv3D.Builder().setKernel(new Shape(2, 2, 2)).setNumFilters(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input =
                        manager.create(
                                new float[] {
                                    9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4, 4, 9, 7, 5,
                                    11, 2, 5, 13, 10, 8, 4
                                },
                                new Shape(1, 1, 3, 3, 3));
                NDArray expected =
                        manager.create(
                                new float[] {61, 42, 55, 49, 56, 60, 57, 62},
                                new Shape(1, 1, 2, 2, 2));

                NDArray out = trainer.forward(new NDList(input)).get(0);
                Assertions.assertEquals(expected, out);

                Shape outputShape = block.getOutputShape(new Shape(1, 1, 3, 3, 3));
                Assertions.assertTrue(out.getShape().equals(outputShape));

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testRNNTanh() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, true);

        Block block =
                new RNN.Builder()
                        .setStateSize(5)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.TANH)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.arange(0, 48, 1).reshape(new Shape(3, 4, 4));
                NDList outputs = trainer.forward(new NDList(input));
                NDArray out = outputs.get(0);
                Assertions.assertEquals(manager.ones(new Shape(3, 4, 5)), out);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testRNNRelu() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        Block block =
                new RNN.Builder()
                        .setStateSize(5)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));
                NDList outputs = trainer.forward(new NDList(input));
                NDArray out = outputs.get(0);
                NDArray expected =
                        manager.create(
                                new float[] {13, 13, 13, 13, 13, 29, 29, 29, 29, 29},
                                new Shape(1, 2, 5));
                Assertions.assertEquals(expected, out);

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testLstm() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        Block block =
                new LSTM.Builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();
                NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));
                NDList outputs = trainer.forward(new NDList(input));
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

                testEncode(manager, block);
            }
        }
    }

    @RunAsTest
    public void testGRU() throws FailedTestException, IOException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, false);

        GRU block =
                new GRU.Builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                NDManager manager = trainer.getManager();

                NDArray input = manager.arange(0, 8, 1).reshape(new Shape(1, 2, 4));

                NDList outputs = trainer.forward(new NDList(input));
                NDArray out = outputs.get(0);
                Assertions.assertAlmostEquals(manager.ones(new Shape(1, 2, 4)), out);

                testEncode(manager, block);
            }
        }
    }

    private void testEncode(NDManager manager, Block block)
            throws IOException, FailedTestException {
        PairList<String, Parameter> original = block.getParameters();
        File temp = File.createTempFile("block", ".param");
        DataOutputStream os = new DataOutputStream(Files.newOutputStream(temp.toPath()));
        block.saveParameters(os);
        block.loadParameters(manager, new DataInputStream(Files.newInputStream(temp.toPath())));
        Files.delete(temp.toPath());
        PairList<String, Parameter> loaded = block.getParameters();
        int bound = original.size();
        for (int idx = 0; idx < bound; idx++) {
            Assertions.assertEquals(original.valueAt(idx), loaded.valueAt(idx));
        }
    }
}
