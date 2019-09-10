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
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.nn.convolutional.Conv1D;
import software.amazon.ai.nn.convolutional.Conv2D;
import software.amazon.ai.nn.convolutional.Conv3D;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.nn.core.Prelu;
import software.amazon.ai.nn.norm.BatchNorm;
import software.amazon.ai.nn.norm.Dropout;
import software.amazon.ai.nn.recurrent.GRU;
import software.amazon.ai.nn.recurrent.LSTM;
import software.amazon.ai.nn.recurrent.RNN;
import software.amazon.ai.training.initializer.NormalInitializer;
import software.amazon.ai.util.PairList;

public class BlockEncodeDecodeTest {
    public static void main(String[] args) {
        String[] cmd = {"-c", BlockEncodeDecodeTest.class.getName()};
        new IntegrationTest()
                .runTests(
                        Stream.concat(Arrays.stream(cmd), Arrays.stream(args))
                                .toArray(String[]::new));
    }

    private void testBlock(Block block, NDArray input) throws IOException, FailedTestException {
        block.forward(new NDList(input));
        PairList<String, Parameter> original = block.getParameters();
        File temp = File.createTempFile("block", ".param");
        DataOutputStream os = new DataOutputStream(Files.newOutputStream(temp.toPath()));
        block.saveParameters(os);
        block.loadParameters(new DataInputStream(Files.newInputStream(temp.toPath())));
        Files.delete(temp.toPath());
        PairList<String, Parameter> loaded = block.getParameters();
        int bound = original.size();
        for (int idx = 0; idx < bound; idx++) {
            Assertions.assertEquals(original.valueAt(idx), loaded.valueAt(idx));
        }
    }

    @RunAsTest
    public void testRnn() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            RNN rnn =
                    new RNN.Builder()
                            .setFactory(factory)
                            .setNumStackedLayers(1)
                            .setStateSize(1)
                            .build();
            rnn.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(rnn, input);
        }
    }

    @RunAsTest
    public void testLstm() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            LSTM lstm =
                    new LSTM.Builder()
                            .setFactory(factory)
                            .setNumStackedLayers(1)
                            .setStateSize(1)
                            .build();
            lstm.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(lstm, input);
        }
    }

    @RunAsTest
    public void testGru() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            GRU gru =
                    new GRU.Builder()
                            .setFactory(factory)
                            .setNumStackedLayers(1)
                            .setStateSize(1)
                            .build();
            gru.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(gru, input);
        }
    }

    @RunAsTest
    public void testDropout() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Dropout dropout = new Dropout.Builder().setFactory(factory).build();
            dropout.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(dropout, input);
        }
    }

    @RunAsTest
    public void testBatchNorm() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            BatchNorm bn = new BatchNorm.Builder().setFactory(factory).build();
            bn.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(bn, input);
        }
    }

    @RunAsTest
    public void testPrelu() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Prelu blk = factory.createPrelu();
            blk.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(blk, input);
        }
    }

    @RunAsTest
    public void testLinear() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Linear blk = new Linear.Builder().setFactory(factory).setOutChannels(10).build();
            blk.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(10, 20));
            testBlock(blk, input);
        }
    }

    @RunAsTest
    public void testConv1D() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Conv1D blk =
                    new Conv1D.Builder()
                            .setFactory(factory)
                            .setKernel(new Shape(1))
                            .setNumFilters(1)
                            .build();
            blk.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(3, 2, 2));
            testBlock(blk, input);
        }
    }

    @RunAsTest
    public void testConv2D() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Conv2D blk =
                    new Conv2D.Builder()
                            .setFactory(factory)
                            .setKernel(new Shape(1, 3))
                            .setNumFilters(1)
                            .build();
            blk.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(1, 3, 4, 4));
            testBlock(blk, input);
        }
    }

    @RunAsTest
    public void testConv3D() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            Conv3D blk =
                    new Conv3D.Builder()
                            .setFactory(factory)
                            .setKernel(new Shape(1, 3, 2))
                            .setNumFilters(1)
                            .build();
            blk.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(1, 3, 2, 4, 4));
            testBlock(blk, input);
        }
    }

    @RunAsTest
    public void testSequentialBlock() throws FailedTestException, IOException {
        try (Model model = Model.newInstance()) {
            BlockFactory factory = model.getBlockFactory();
            SequentialBlock mlp = factory.createSequential();
            mlp.add(new Linear.Builder().setFactory(factory).setOutChannels(128).build());
            mlp.add(factory.activation().reluBlock());
            mlp.add(new Dropout.Builder().setFactory(factory).setProbability(0.6f).build());
            mlp.add(new Linear.Builder().setFactory(factory).setOutChannels(64).build());
            mlp.add(factory.activation().reluBlock());
            mlp.add(new Dropout.Builder().setFactory(factory).setProbability(0.9f).build());
            mlp.add(new Linear.Builder().setFactory(factory).setOutChannels(10).build());
            mlp.setInitializer(new NormalInitializer());
            NDArray input = model.getNDManager().ones(new Shape(32, 784));
            testBlock(mlp, input);
        }
    }
}
