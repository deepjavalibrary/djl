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
package ai.djl.integration.tests.model_zoo.classification;

import ai.djl.Model;
import ai.djl.integration.util.Assertions;
import ai.djl.modality.Classification;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.repository.Repository;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Nag;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.util.PairList;
import ai.djl.zoo.ModelNotFoundException;
import ai.djl.zoo.ZooModel;
import ai.djl.zoo.cv.classification.ResNetModelLoader;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ResnetTest {

    @Test
    public void testTrain() {
        Optimizer optimizer =
                new Nag.Builder()
                        .setRescaleGrad(1.0f / 100)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .setMomentum(0.9f)
                        .build();

        TrainingConfig config =
                new DefaultTrainingConfig(Initializer.ONES)
                        .setOptimizer(optimizer)
                        .setLoss(Loss.softmaxCrossEntropyLoss());

        Block resNet50 =
                new ResNetV1.Builder()
                        .setImageShape(new Shape(1, 28, 28))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();

        try (Model model = Model.newInstance()) {
            model.setBlock(resNet50);

            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(100, 1, 28, 28);
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                NDManager manager = trainer.getManager();

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(100, 1));
                Batch batch = new Batch(manager, new NDList(input), new NDList(label));
                PairList<String, Parameter> parameters = resNet50.getParameters();
                trainer.train(batch);
                trainer.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.ones(new Shape(16)).muli(.8577);
                NDArray expectedAtIndex87 = manager.ones(new Shape(32, 32, 3, 3));
                Assert.assertEquals(parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(1).getValue().getArray(), expectedAtIndex1);
                Assert.assertEquals(expectedAtIndex87, parameters.get(87).getValue().getArray());
            }
        }
    }

    @Test
    public void testLoad() throws IOException, ModelNotFoundException {
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("numLayers", "50");
        criteria.put("dataset", "cifar10");
        // TODO: Train real model zoo models and convert this test to use instead of sample model
        //        try (ZooModel<BufferedImage, List<Classification>> model =
        //                ModelZoo.RESNET.loadModel(criteria)) {
        try (ZooModel<BufferedImage, Classification> model =
                new ResNetModelLoader(Repository.newInstance("test", "src/main/resources/repo"))
                        .loadModel(criteria)) {
            TrainingConfig config =
                    new DefaultTrainingConfig(Initializer.ONES).setLoss(Loss.l1Loss());
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(16, 3, 32, 32);

                trainer.initialize(new DataDesc[] {new DataDesc(inputShape, DataType.FLOAT32)});

                NDManager manager = trainer.getManager();
                Shape[] outputShape =
                        model.getBlock().getOutputShapes(manager, new Shape[] {inputShape});

                NDArray data = manager.ones(new Shape(16, 3, 32, 32));
                NDArray label = manager.ones(outputShape[0]);
                Batch batch = new Batch(manager, new NDList(data), new NDList(label));
                trainer.train(batch);
            }
        }
    }
}
