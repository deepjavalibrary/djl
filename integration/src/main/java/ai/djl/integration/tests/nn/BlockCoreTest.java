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
package ai.djl.integration.tests.nn;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.integration.util.Assertions;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv1D;
import ai.djl.nn.convolutional.Conv2D;
import ai.djl.nn.convolutional.Conv3D;
import ai.djl.nn.core.Embedding;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.recurrent.GRU;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.nn.recurrent.RNN;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BlockCoreTest {

    @Test
    public void testLinear() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        long outSize = 3;
        Block block = new Linear.Builder().setOutChannels(outSize).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2, 2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected =
                        data.dot(manager.ones(new Shape(outSize, 2)).transpose())
                                .add(manager.zeros(new Shape(2, outSize)));
                Assert.assertEquals(result, expected);

                testEncode(manager, block);
            }
        }

        block = new Linear.Builder().setOutChannels(outSize).optBias(false).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2, 2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected = data.dot(manager.ones(new Shape(outSize, 2)).transpose());
                Assert.assertEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testLinearWithDefinedLayout() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        long outSize = 3;
        Block block = new Linear.Builder().setOutChannels(outSize).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape =
                        new Shape(
                                new long[] {2, 2},
                                new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL});
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected =
                        data.dot(manager.ones(new Shape(outSize, 2)).transpose())
                                .add(manager.zeros(new Shape(2, outSize)));
                Assert.assertEquals(result, expected);

                testEncode(manager, block);
            }
        }

        block = new Linear.Builder().setOutChannels(outSize).optBias(false).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape =
                        new Shape(
                                new long[] {2, 2},
                                new LayoutType[] {LayoutType.BATCH, LayoutType.CHANNEL});
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected = data.dot(manager.ones(new Shape(outSize, 2)).transpose());
                Assert.assertEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testBatchNorm() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block = new BatchNorm.Builder().optAxis(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2, 2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray expected = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testDropout() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block = new Dropout.Builder().optProbability(.5f).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2, 2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertTrue(result.lte(result).all().getBoolean());

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testEmbedding() {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Embedding<Character> block =
                new Embedding.Builder<Character>()
                        .setItems(Arrays.asList('a', 'b', 'c'))
                        .setEmbeddingSize(2)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);
            model.setDataType(DataType.INT32);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                ParameterStore parameterStore = new ParameterStore(manager, false);

                // TODO: use trainer.forward
                Assert.assertEquals(
                        block.forward(parameterStore, manager, 'x'),
                        manager.create(new int[] {1, 1}));
                Assert.assertEquals(
                        block.forward(parameterStore, manager, new Character[] {'a', 'b'}),
                        manager.create(new int[] {1, 1, 1, 1}, new Shape(2, 2)));
            }
        }
    }

    @Test
    public void testConv1D() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block =
                new Conv1D.Builder()
                        .setKernel(new Shape(2))
                        .setNumFilters(1)
                        .optBias(false)
                        .build();

        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 4, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data =
                        manager.create(
                                new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                                inputShape);
                NDArray expected = manager.create(new float[] {61, 55, 44}, new Shape(1, 1, 3));
                NDArray out = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(out, expected);

                Shape[] outputShape = block.getOutputShapes(manager, new Shape[] {inputShape});
                Assert.assertEquals(out.getShape(), outputShape[0]);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv2D() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block = new Conv2D.Builder().setKernel(new Shape(2, 2)).setNumFilters(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 1, 4, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data =
                        manager.create(
                                new float[] {9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4},
                                inputShape);
                NDArray expected =
                        manager.create(
                                new float[] {22, 24, 25, 21, 26, 23, 39, 31, 19},
                                new Shape(1, 1, 3, 3));

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv3D() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block = new Conv3D.Builder().setKernel(new Shape(2, 2, 2)).setNumFilters(1).build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 1, 3, 3, 3);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data =
                        manager.create(
                                new float[] {
                                    9, 8, 3, 6, 1, 4, 9, 7, 5, 11, 2, 5, 13, 10, 8, 4, 4, 9, 7, 5,
                                    11, 2, 5, 13, 10, 8, 4
                                },
                                inputShape);
                NDArray expected =
                        manager.create(
                                new float[] {60, 41, 54, 48, 55, 59, 56, 61},
                                new Shape(1, 1, 2, 2, 2));

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result, expected);

                Shape[] outputShape =
                        block.getOutputShapes(manager, new Shape[] {new Shape(1, 1, 3, 3, 3)});
                Assert.assertEquals(result.getShape(), outputShape[0]);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testRNNTanh() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block =
                new RNN.Builder()
                        .setStateSize(5)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.TANH)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(3, 4, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.arange(0, 48, 1).reshape(inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, manager.ones(new Shape(3, 4, 5)));

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testRNNRelu() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block =
                new RNN.Builder()
                        .setStateSize(5)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.arange(0, 8, 1).reshape(inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected =
                        manager.create(
                                new float[] {7, 12, 12, 12, 12, 23, 28, 28, 28, 28},
                                new Shape(1, 2, 5));
                Assert.assertEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testLstm() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        Block block =
                new LSTM.Builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.arange(0, 8, 1).reshape(inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                NDArray expected =
                        manager.create(
                                new float[] {
                                    0.9638f, 0.9631f, 0.9576f, 0.964f, 0.9639f, 0.964f, 0.9576f,
                                    0.964f
                                },
                                new Shape(1, 2, 4));
                Assertions.assertAlmostEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testGRU() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());

        GRU block =
                new GRU.Builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.arange(0, 8, 1).reshape(inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, manager.ones(new Shape(1, 2, 4)));

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testSequentialBlock() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());
        SequentialBlock block = new SequentialBlock();
        block.add(x -> new NDList(x.singletonOrThrow().mul(6.5f)));
        block.add(new Linear.Builder().setOutChannels(10).build());
        block.add(new Linear.Builder().setOutChannels(5).build());

        Assert.assertEquals(block.getChildren().size(), 3);
        Assert.assertEquals(block.getDirectParameters().size(), 0);
        Assert.assertEquals(block.getParameters().size(), 4);

        block.addAll(
                Arrays.asList(
                        new Linear.Builder().setOutChannels(3).build(),
                        new LambdaBlock(x -> new NDList(x.singletonOrThrow().div(2f)))));
        Assert.assertEquals(block.getChildren().size(), 5);
        Assert.assertEquals(block.getParameters().size(), 6);

        block.removeLastBlock();
        Assert.assertEquals(block.getChildren().size(), 4);

        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 3);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(1, 3));
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(
                        result, manager.create(new float[] {975, 975, 975}, new Shape(1, 3)));

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testParallelBlock() throws IOException, MalformedModelException {
        TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l2Loss());
        ParallelBlock block =
                new ParallelBlock(
                        list ->
                                new NDList(
                                        list.get(0).singletonOrThrow(),
                                        list.get(1).singletonOrThrow(),
                                        list.get(2).singletonOrThrow()));
        block.add(new Linear.Builder().setOutChannels(3).build());
        block.add(x -> new NDList(x.singletonOrThrow().sum()));
        block.add(new Linear.Builder().setOutChannels(2).build());

        Assert.assertEquals(block.getChildren().size(), 3);
        Assert.assertEquals(block.getDirectParameters().size(), 0);
        Assert.assertEquals(block.getParameters().size(), 4);

        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 3);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(new Shape(1, 3));
                NDList results = trainer.forward(new NDList(data));
                Assertions.assertAlmostEquals(
                        results.get(0), manager.create(new float[] {3, 3, 3}, new Shape(1, 3)));
                Assertions.assertAlmostEquals(results.get(1), manager.create(3));
                Assertions.assertAlmostEquals(
                        results.get(2), manager.create(new float[] {3, 3}, new Shape(1, 2)));

                testEncode(manager, block);
            }
        }
    }

    private void testEncode(NDManager manager, Block block)
            throws IOException, MalformedModelException {
        PairList<String, Parameter> original = block.getParameters();
        File temp = File.createTempFile("block", ".param");
        DataOutputStream os = new DataOutputStream(Files.newOutputStream(temp.toPath()));
        block.saveParameters(os);
        block.loadParameters(manager, new DataInputStream(Files.newInputStream(temp.toPath())));
        Files.delete(temp.toPath());
        PairList<String, Parameter> loaded = block.getParameters();
        int bound = original.size();
        for (int idx = 0; idx < bound; idx++) {
            Assert.assertEquals(original.valueAt(idx), loaded.valueAt(idx));
        }
    }
}
