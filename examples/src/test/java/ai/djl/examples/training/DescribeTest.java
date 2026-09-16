/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.training;

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.basicmodelzoo.cv.classification.ResNetV2;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

public class DescribeTest {

    @Test
    public void testDescribe() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        // Construct neural network
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[] {128, 64});

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Path path = Paths.get("src/test/resources/describe/Mnist.txt");

                try (InputStream is = Files.newInputStream(path)) {
                    List<String> expected = Utils.readLines(is);
                    List<String> actual =
                            Arrays.asList(
                                    Blocks.describe(
                                                    trainer.getModel().getBlock(),
                                                    "SequentialBlock",
                                                    0)
                                            .split("\\R"));
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }

    @Test
    public void testDescribeInitialized() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        // Construct neural network
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[] {128, 64});

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);
                trainer.initialize(inputShape);

                Path path = Paths.get("src/test/resources/describe/MnistInitialized.txt");

                try (InputStream is = Files.newInputStream(path)) {
                    List<String> expected = Utils.readLines(is);
                    List<String> actual =
                            Arrays.asList(
                                    Blocks.describe(
                                                    trainer.getModel().getBlock(),
                                                    "SequentialBlock",
                                                    0)
                                            .split("\\R"));
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }

    @Test
    public void testKerasSequentialApi() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        // Construct neural network
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[] {128, 64});

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Path path = Paths.get("src/test/resources/describe/MnistSequential.py");

                try (InputStream is = Files.newInputStream(path)) {
                    List<String> expected = Utils.readLines(is);
                    List<String> actual =
                            Arrays.asList(Blocks.describeAsTensorflow(trainer, false).split("\\R"));
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }

    @Test
    public void testKerasFunctionalApi() throws IOException {
        TrainingConfig config = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

        // Construct neural network
        Block block =
                new Mlp(
                        Mnist.IMAGE_HEIGHT * Mnist.IMAGE_WIDTH,
                        Mnist.NUM_CLASSES,
                        new int[] {128, 64});

        try (Model model = Model.newInstance("mlp")) {
            model.setBlock(block);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(1, Mnist.IMAGE_HEIGHT, Mnist.IMAGE_WIDTH);
                trainer.initialize(inputShape);

                Path path = Paths.get("src/test/resources/describe/MnistFunctional.py");

                try (InputStream is = Files.newInputStream(path)) {
                    List<String> expected = Utils.readLines(is);
                    List<String> actual =
                            Arrays.asList(Blocks.describeAsTensorflow(trainer, true).split("\\R"));
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }

    @Test
    public void testKerasFunctionalApiForResnet() throws IOException {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block resNet50 =
                ResNetV2.builder()
                        .setImageShape(new Shape(3, 32, 32))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();
        try (Model model = Model.newInstance("resnet")) {
            model.setBlock(resNet50);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 3, 32, 32);
                trainer.initialize(inputShape);

                Path path = Paths.get("src/test/resources/describe/Resnet50.py");

                try (InputStream is = Files.newInputStream(path)) {
                    List<String> expected = Utils.readLines(is);
                    List<String> actual =
                            Arrays.asList(Blocks.describeAsTensorflow(trainer, true).split("\\R"));
                    Assert.assertEquals(actual, expected);
                }
            }
        }
    }
}
