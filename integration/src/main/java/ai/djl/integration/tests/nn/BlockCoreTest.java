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
import ai.djl.nn.ParameterList;
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
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
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
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        long outSize = 3;
        Block block = Linear.builder().setOutChannels(outSize).build();
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

        block = Linear.builder().setOutChannels(outSize).optBias(false).build();
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        long outSize = 3;
        Block block = Linear.builder().setOutChannels(outSize).build();
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

        block = Linear.builder().setOutChannels(outSize).optBias(false).build();
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Block block = BatchNorm.builder().optAxis(1).build();
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Block block = Dropout.builder().optProbability(.5f).build();
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Embedding<Character> block =
                Embedding.builder()
                        .setType(Character.class)
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

                Assert.assertEquals(
                        trainer.forward(new NDList(block.embed(manager, 'x'))).singletonOrThrow(),
                        manager.create(new int[] {1, 1}));

                Assert.assertEquals(
                        trainer.forward(
                                        new NDList(
                                                block.embed(manager, new Character[] {'a', 'b'})))
                                .singletonOrThrow(),
                        manager.create(new int[] {1, 1, 1, 1}, new Shape(2, 2)));
            }
        }
    }

    @Test
    public void testConv1D() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Block block =
                Conv1D.builder().setKernel(new Shape(2)).setNumFilters(1).optBias(false).build();

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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Block block = Conv2D.builder().setKernel(new Shape(2, 2)).setNumFilters(1).build();
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);

        Block block = Conv3D.builder().setKernel(new Shape(2, 2, 2)).setNumFilters(1).build();
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
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(new XavierInitializer())
                        .optDevices(getDevices());
        Block block =
                RNN.builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.TANH)
                        .optStateOutput(true)
                        .build();
        try (Model model = Model.newInstance(config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                Engine.getInstance().setRandomSeed(1234);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.randomUniform(0, 10, inputShape);
                NDArray labels = manager.randomUniform(0, 1, inputShape);
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        0.9411f, 1f, 1f, 0.918f, -0.9883f, 1f, 1f, -0.4683f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    Assertions.assertAlmostEquals(result.size(), 2);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    Assertions.assertAlmostEquals(lossValue.getFloat(), -1.14439737);

                    collector.backward(lossValue);
                    ParameterList parameterList = model.getBlock().getParameters();
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray(),
                            new Shape(4, 4),
                            3.059284f,
                            0.19120525f,
                            0.8596075f,
                            -0.8024095f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray().getGradient(),
                            new Shape(4, 4),
                            -1.7775341f,
                            -0.11109588f,
                            -7.650868E-9f,
                            -0.5153482f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray(),
                            new Shape(4, 4),
                            -1.9064846f,
                            -0.11915529f,
                            0.84814364f,
                            -0.7939344f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray().getGradient(),
                            new Shape(4, 4),
                            -0.01559069f,
                            -9.7441813E-4f,
                            0.0f,
                            -0.003516526f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray(),
                            new Shape(4),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray().getGradient(),
                            new Shape(4),
                            -0.07112471f,
                            -0.017781178f,
                            -9.5081925E-9f,
                            -0.055576526f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray(),
                            new Shape(4),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray().getGradient(),
                            new Shape(4),
                            -0.07112471f,
                            -0.017781178f,
                            -9.5081925E-9f,
                            -0.055576526f);
                }
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testRNNRelu() throws IOException, MalformedModelException {
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(new XavierInitializer())
                        .optDevices(getDevices());
        Block block =
                RNN.builder()
                        .setStateSize(4)
                        .setNumStackedLayers(1)
                        .setActivation(RNN.Activation.RELU)
                        .build();
        try (Model model = Model.newInstance(config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                Engine.getInstance().setRandomSeed(1234);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.randomUniform(0, 10, inputShape);
                NDArray labels = manager.randomUniform(0, 1, inputShape);
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        1.7478f, 8.0788f, 8.8983f, 1.5759f, 0f, 10.8066f, 4.7126f,
                                        0f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.singletonOrThrow(), expected);
                    NDArray lossValue = loss.evaluate(new NDList(labels), result);
                    Assertions.assertAlmostEquals(lossValue.getFloat(), -6.27057);

                    collector.backward(lossValue);
                    ParameterList parameterList = model.getBlock().getParameters();
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray(),
                            new Shape(4, 4),
                            3.059284f,
                            0.19120525f,
                            0.8596075f,
                            -0.8024095f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray().getGradient(),
                            new Shape(4, 4),
                            -22.968185f,
                            -1.4355116f,
                            -0.02831553f,
                            -3.8558674f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray(),
                            new Shape(4, 4),
                            -1.9064846f,
                            -0.11915529f,
                            0.84814364f,
                            -0.7939344f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray().getGradient(),
                            new Shape(4, 4),
                            -7.3317056f,
                            -0.4582316f,
                            0f,
                            -2.2465506f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray(),
                            new Shape(4),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray().getGradient(),
                            new Shape(4),
                            -0.9217882f,
                            -0.23044705f,
                            -0.035189405f,
                            -0.43959588f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray(),
                            new Shape(4),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray().getGradient(),
                            new Shape(4),
                            -0.9217882f,
                            -0.23044705f,
                            -0.035189405f,
                            -0.43959588f);
                }
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testLstm() throws IOException, MalformedModelException {
        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(new XavierInitializer())
                        .optDevices(getDevices());
        Block block =
                LSTM.builder().setStateSize(4).setNumStackedLayers(1).optStateOutput(true).build();
        try (Model model = Model.newInstance(config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                Engine.getInstance().setRandomSeed(1234);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.randomUniform(0, 10, inputShape);
                NDArray labels = manager.randomUniform(0, 1, inputShape);
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        0.0028f, 0.0253f, -0.0273f, -0.0095f, 0.3141f, 0.5168f,
                                        -0.0266f, -0.6412f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.head(), expected);
                    NDArray lossValue =
                            loss.evaluate(new NDList(labels), new NDList(result.head()));
                    Assertions.assertAlmostEquals(lossValue.getFloat(), -0.03504385f);

                    collector.backward(lossValue);
                    ParameterList parameterList = model.getBlock().getParameters();
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray(),
                            new Shape(16, 4),
                            -0.23054221f,
                            -0.003602222f,
                            0.5436635f,
                            -0.537854f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray().getGradient(),
                            new Shape(16, 4),
                            -0.48049742f,
                            -0.007507772f,
                            0.0771877f,
                            -0.22537863f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray(),
                            new Shape(16, 4),
                            1.2113954f,
                            0.018928053f,
                            0.5252731f,
                            -0.54756975f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray().getGradient(),
                            new Shape(16, 4),
                            1.9402013E-4f,
                            3.0315646E-6f,
                            6.4684724E-4f,
                            -6.010107E-4f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray(),
                            new Shape(16),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray().getGradient(),
                            new Shape(16),
                            -0.017782848f,
                            -0.001111428f,
                            0.008951807f,
                            -0.027755652f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray(),
                            new Shape(16),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray().getGradient(),
                            new Shape(16),
                            -0.017782848f,
                            -0.001111428f,
                            0.008951807f,
                            -0.027755652f);
                }
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testGRU() throws IOException, MalformedModelException {

        Loss loss = new SoftmaxCrossEntropyLoss("SmCeLoss", 1, -1, false, true);
        TrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optInitializer(new XavierInitializer())
                        .optDevices(getDevices());
        GRU block = GRU.builder().setStateSize(4).setNumStackedLayers(1).build();
        try (Model model = Model.newInstance(config.getDevices()[0])) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, 2, 4);
                Engine.getInstance().setRandomSeed(1234);
                trainer.initialize(inputShape);
                NDManager manager = trainer.getManager();
                NDArray data = manager.randomUniform(0, 10, inputShape);
                NDArray labels = manager.randomUniform(0, 1, inputShape);
                try (GradientCollector collector = trainer.newGradientCollector()) {
                    NDList result = trainer.forward(new NDList(data));
                    NDArray expected =
                            manager.create(
                                    new float[] {
                                        0.9973f, 0.6594f, -0.886f, -0.9176f, -0.0989f, 0.437f,
                                        0.2401f, -0.933f
                                    },
                                    new Shape(1, 2, 4));
                    Assertions.assertAlmostEquals(result.singletonOrThrow(), expected);
                    NDArray lossValue = loss.evaluate(new NDList(labels), result);
                    Assertions.assertAlmostEquals(lossValue.getFloat(), -0.104264974f);

                    collector.backward(lossValue);
                    ParameterList parameterList = model.getBlock().getParameters();
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray(),
                            new Shape(12, 4),
                            0.12911546f,
                            0.0026899055f,
                            0.6078343f,
                            -0.6013391f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(0).getValue().getArray().getGradient(),
                            new Shape(12, 4),
                            -10.557455f,
                            -0.21994698f,
                            0.18836471f,
                            -2.2442424f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray(),
                            new Shape(12, 4),
                            -0.085866004f,
                            -0.0017888751f,
                            0.5796961f,
                            -0.6122016f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(1).getValue().getArray().getGradient(),
                            new Shape(12, 4),
                            0.06644759f,
                            0.0013843249f,
                            0.2643293f,
                            -0.2872954f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray(),
                            new Shape(12),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(2).getValue().getArray().getGradient(),
                            new Shape(12),
                            -0.64488584f,
                            -0.053740487f,
                            0.023915248f,
                            -0.31230754f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray(),
                            new Shape(12),
                            0f,
                            0f,
                            0f,
                            0f);
                    TestUtils.verifyNDArrayValues(
                            parameterList.get(3).getValue().getArray().getGradient(),
                            new Shape(12),
                            -0.5020572f,
                            -0.0418381f,
                            0.023915248f,
                            -0.3114909f);
                }
                testEncode(manager, block);
            }
        }
    }

    @Test
    public void testSequentialBlock() throws IOException, MalformedModelException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);
        SequentialBlock block = new SequentialBlock();
        block.add(x -> new NDList(x.singletonOrThrow().mul(6.5f)));
        block.add(Linear.builder().setOutChannels(10).build());
        block.add(Linear.builder().setOutChannels(5).build());

        Assert.assertEquals(block.getChildren().size(), 3);
        Assert.assertEquals(block.getDirectParameters().size(), 0);
        Assert.assertEquals(block.getParameters().size(), 4);

        block.addAll(
                Arrays.asList(
                        Linear.builder().setOutChannels(3).build(),
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
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.l2Loss()).optInitializer(Initializer.ONES);
        ParallelBlock block =
                new ParallelBlock(
                        list ->
                                new NDList(
                                        list.get(0).singletonOrThrow(),
                                        list.get(1).singletonOrThrow(),
                                        list.get(2).singletonOrThrow()));
        block.add(Linear.builder().setOutChannels(3).build());
        block.add(x -> new NDList(x.singletonOrThrow().sum()));
        block.add(Linear.builder().setOutChannels(2).build());

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

    private static Device[] getDevices() {
        if (TestUtils.isWindows() && TestUtils.isMxnet()) {
            return new Device[] {
                Device.cpu()
            }; // TODO: RNN is not implemented on MXNet without cuDNN
        }
        return Device.getDevices(1);
    }
}
