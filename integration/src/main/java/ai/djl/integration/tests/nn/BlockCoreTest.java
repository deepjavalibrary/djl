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

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.integration.util.TestUtils;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv1d;
import ai.djl.nn.convolutional.Conv1dTranspose;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.convolutional.Conv2dTranspose;
import ai.djl.nn.convolutional.Conv3d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.recurrent.GRU;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.nn.recurrent.RNN;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        long outSize = 3;
        Block block = Linear.builder().setUnits(outSize).build();
        try (Model model = Model.newInstance("model")) {
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

        block = Linear.builder().setUnits(outSize).optBias(false).build();
        try (Model model = Model.newInstance("model")) {
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

        outSize = 10;
        block = Linear.builder().setUnits(outSize).build();
        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(10, 20, 12);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.ones(inputShape);
                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(result.getShape(), new Shape(10, 20, 10));
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testLinearWithDefinedLayout() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        long outSize = 3;
        Block block = Linear.builder().setUnits(outSize).build();
        try (Model model = Model.newInstance("model")) {
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

        block = Linear.builder().setUnits(outSize).optBias(false).build();
        try (Model model = Model.newInstance("model")) {
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

    @SuppressWarnings("try")
    @Test
    public void testBatchNorm() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = BatchNorm.builder().build();
        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for BatchNorm to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(2, 2);
                    trainer.initialize(inputShape);

                    NDManager manager = trainer.getManager();
                    NDArray data = manager.create(new float[] {1, 2, 3, 4}, inputShape);
                    NDArray expected = manager.create(new float[] {-1, -1, 1, 1}, inputShape);
                    NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                    Assertions.assertAlmostEquals(result, expected);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testLayerNorm() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = LayerNorm.builder().build();
        try (Model model = Model.newInstance("model", Device.cpu(), "PyTorch")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(2, 2);
                    trainer.initialize(inputShape);

                    NDManager manager = trainer.getManager();
                    NDArray data = manager.create(new float[] {1, 3, 2, 4}, inputShape);
                    NDArray expected = manager.create(new float[] {-1, 1, -1, 1}, inputShape);
                    NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                    Assertions.assertAlmostEquals(result, expected);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void test2LayerNorm() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = LayerNorm.builder().axis(2, 3).build();
        try (Model model = Model.newInstance("model", Device.cpu(), "PyTorch")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(1, 2, 1, 2);
                    trainer.initialize(inputShape);

                    NDManager manager = trainer.getManager();
                    NDArray data = manager.create(new float[] {1, 3, 2, 4}, inputShape);
                    NDArray expected = manager.create(new float[] {-1, 1, -1, 1}, inputShape);
                    NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                    Assertions.assertAlmostEquals(result, expected);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testDropout() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = Dropout.builder().optRate(.5f).build();
        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for Dropout to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
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
    }

    @Test
    public void testEmbedding() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        TrainableWordEmbedding block =
                TrainableWordEmbedding.builder()
                        .setVocabulary(new SimpleVocabulary(Arrays.asList("a", "b", "c")))
                        .setEmbeddingSize(2)
                        .build();
        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(2);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();

                Assert.assertEquals(
                        trainer.forward(new NDList(manager.create(block.embed("x"))))
                                .singletonOrThrow(),
                        manager.create(new float[] {1, 1}));

                Assert.assertEquals(
                        trainer.forward(new NDList(block.embed(manager, new String[] {"x", "y"})))
                                .singletonOrThrow(),
                        manager.create(new float[] {1, 1, 1, 1}, new Shape(2, 2)));
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv1d() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block =
                Conv1d.builder().setKernelShape(new Shape(2)).setFilters(1).optBias(false).build();

        try (Model model = Model.newInstance("model")) {
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

                Shape[] outputShape = block.getOutputShapes(new Shape[] {inputShape});
                Assert.assertEquals(out.getShape(), outputShape[0]);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv1dTranspose() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block =
                Conv1dTranspose.builder()
                        .setKernelShape(new Shape(2))
                        .setFilters(1)
                        .optBias(false)
                        .build();

        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 1, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data = manager.create(new float[] {19, 84, 20, 10}, inputShape);
                NDArray expected =
                        manager.create(new float[] {19, 103, 104, 30, 10}, new Shape(1, 1, 5));
                NDArray out = trainer.forward(new NDList(data)).singletonOrThrow();
                Assert.assertEquals(out, expected);

                Shape[] outputShape = block.getOutputShapes(new Shape[] {inputShape});
                Assert.assertEquals(out.getShape(), outputShape[0]);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv2d() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = Conv2d.builder().setKernelShape(new Shape(2, 2)).setFilters(1).build();
        try (Model model = Model.newInstance("model")) {
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
    public void testConv2dTranspose() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block =
                Conv2dTranspose.builder().setKernelShape(new Shape(2, 2)).setFilters(1).build();
        try (Model model = Model.newInstance("model")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 1, 4, 4);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                NDArray data =
                        manager.create(
                                new float[] {
                                    19, 84, 20, 10, 4, 24, 22, 10, 3, 3, 8, 12, 36, 32, 6, 2
                                },
                                inputShape);
                NDArray expected =
                        manager.create(
                                new float[] {
                                    19, 103, 104, 30, 10, 23, 131, 150, 62, 20, 7, 34, 57, 52, 22,
                                    39, 74, 49, 28, 14, 36, 68, 38, 8, 2
                                },
                                new Shape(1, 1, 5, 5));

                NDArray result = trainer.forward(new NDList(data)).singletonOrThrow();
                Assertions.assertAlmostEquals(result, expected);

                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testConv3d() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block block = Conv3d.builder().setKernelShape(new Shape(2, 2, 2)).setFilters(1).build();
        try (Model model = Model.newInstance("model")) {
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

                Shape[] outputShape = block.getOutputShapes(new Shape[] {new Shape(1, 1, 3, 3, 3)});
                Assert.assertEquals(result.getShape(), outputShape[0]);

                testEncode(manager, block);
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testRNNTanh() throws IOException, MalformedModelException {
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optDevices(TestUtils.getDevices());
        Block block =
                RNN.builder()
                        .setStateSize(4)
                        .setNumLayers(1)
                        .setActivation(RNN.Activation.TANH)
                        .optBatchFirst(true)
                        .optReturnState(true)
                        .build();
        try (Model model = Model.newInstance("model", config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for BatchNorm to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(1, 2, 4);
                    Engine.getInstance().setRandomSeed(1234);
                    trainer.initialize(inputShape);
                    NDManager manager = trainer.getManager();
                    NDArray data =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDArray labels =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {1, 1, 1, 1, 1, 1, 1, 1}, new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    Assertions.assertAlmostEquals(result.size(), 2);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    Assertions.assertAlmostEquals(lossValue.getFloat(), 24.9533);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testRNNRelu() throws IOException, MalformedModelException {
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optDevices(TestUtils.getDevices());
        Block block =
                RNN.builder()
                        .setStateSize(4)
                        .setNumLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .optBatchFirst(true)
                        .optReturnState(true)
                        .build();
        try (Model model = Model.newInstance("model", config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for BatchNorm to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(1, 2, 4);
                    Engine.getInstance().setRandomSeed(1234);
                    trainer.initialize(inputShape);
                    NDManager manager = trainer.getManager();
                    NDArray data =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDArray labels =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {10, 10, 10, 10, 66, 66, 66, 66},
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    Assertions.assertAlmostEquals(result.size(), 2);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    // loss should be the same as testRNNTanh because outputs are equal for each
                    // class
                    Assertions.assertAlmostEquals(lossValue.getFloat(), 24.9533);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testLstm() throws IOException, MalformedModelException {
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optDevices(TestUtils.getDevices());
        Block block =
                LSTM.builder()
                        .setStateSize(4)
                        .setNumLayers(1)
                        .optBatchFirst(true)
                        .optReturnState(true)
                        .build();
        try (Model model = Model.newInstance("model", config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for BatchNorm to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(1, 2, 4);
                    Engine.getInstance().setRandomSeed(1234);
                    trainer.initialize(inputShape);
                    NDManager manager = trainer.getManager();
                    NDArray data =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDArray labels =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        00.7615f, 0.7615f, 0.7615f, 0.7615f, 0.964f, 0.964f, 0.964f,
                                        0.964f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    Assertions.assertAlmostEquals(result.size(), 3);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    Assertions.assertAlmostEquals(lossValue.getFloat(), 24.9533);
                    testEncode(manager, block);
                }
            }
        }
    }

    @SuppressWarnings("try")
    @Test
    public void testGRU() throws IOException, MalformedModelException {

        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT)
                        .optDevices(TestUtils.getDevices());
        GRU block =
                GRU.builder()
                        .setStateSize(4)
                        .setNumLayers(1)
                        .optBatchFirst(true)
                        .optReturnState(false)
                        .build();
        try (Model model = Model.newInstance("model", config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                // the unused GradientCollector is for BatchNorm to know it is on training mode
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    Shape inputShape = new Shape(1, 2, 4);
                    Engine.getInstance().setRandomSeed(1234);
                    trainer.initialize(inputShape);
                    NDManager manager = trainer.getManager();
                    NDArray data =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDArray labels =
                            manager.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8})
                                    .reshape(inputShape);
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f,
                                        4.54187393e-05f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    Assertions.assertAlmostEquals(result.size(), 1);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    Assertions.assertAlmostEquals(lossValue.getFloat(), 24.9533);
                    testEncode(manager, block);
                }
            }
        }
    }

    @Test
    public void testSequentialBlock() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        SequentialBlock block = new SequentialBlock();
        block.addSingleton(x -> x.mul(6.5f));
        block.add(Linear.builder().setUnits(10).build());
        block.add(Linear.builder().setUnits(5).build());

        Assert.assertEquals(block.getChildren().size(), 3);
        Assert.assertEquals(block.getDirectParameters().size(), 0);
        Assert.assertEquals(block.getParameters().size(), 4);

        block.addAll(
                Arrays.asList(
                        Linear.builder().setUnits(3).build(),
                        LambdaBlock.singleton(x -> x.div(2f))));
        Assert.assertEquals(block.getChildren().size(), 5);
        Assert.assertEquals(block.getParameters().size(), 6);

        block.removeLastBlock();
        Assert.assertEquals(block.getChildren().size(), 4);

        try (Model model = Model.newInstance("model")) {
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
        ParallelBlock block =
                new ParallelBlock(
                        list ->
                                new NDList(
                                        list.get(0).singletonOrThrow(),
                                        list.get(1).singletonOrThrow(),
                                        list.get(2).singletonOrThrow()));
        block.add(Linear.builder().setUnits(3).build());
        block.add(x -> new NDList(x.singletonOrThrow().sum()));
        block.add(Linear.builder().setUnits(2).build());

        Assert.assertEquals(block.getChildren().size(), 3);
        Assert.assertEquals(block.getDirectParameters().size(), 0);
        Assert.assertEquals(block.getParameters().size(), 4);

        try (Model model = Model.newInstance("model")) {
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
