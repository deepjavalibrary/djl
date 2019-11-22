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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.integration.util.Assertions;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Nag;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.PairList;
import ai.djl.zoo.ModelZoo;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
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
                new DefaultTrainingConfig(Initializer.ONES, Loss.softmaxCrossEntropyLoss())
                        .setOptimizer(optimizer);

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
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(100, 1));
                Batch batch =
                        new Batch(manager, new NDList(input), new NDList(label), Batchifier.STACK);
                PairList<String, Parameter> parameters = resNet50.getParameters();
                trainer.trainBatch(batch);
                trainer.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.ones(new Shape(16, 16, 3, 3));
                NDArray expectedAtIndex87 = manager.ones(new Shape(32));
                Assert.assertEquals(parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(1).getValue().getArray(), expectedAtIndex1);
                Assert.assertEquals(expectedAtIndex87, parameters.get(87).getValue().getArray());
            }
        }
    }

    @Test
    public void testLoadPredict()
            throws IOException, ModelNotFoundException, TranslateException,
                    MalformedModelException {
        try (ZooModel<BufferedImage, Classifications> model = getModel()) {
            try (Predictor<NDList, NDList> predictor = model.newPredictor(new TestTranslator())) {
                NDList input = new NDList(model.getNDManager().ones(new Shape(3, 32, 32)));
                List<NDList> inputs = Collections.nCopies(16, input);
                predictor.batchPredict(inputs);
            }
        }
    }

    @Test
    public void testLoadTrain()
            throws IOException, ModelNotFoundException, MalformedModelException {
        try (ZooModel<BufferedImage, Classifications> model = getModel()) {
            TrainingConfig config = new DefaultTrainingConfig(Initializer.ONES, Loss.l1Loss());
            try (Trainer trainer = model.newTrainer(config)) {
                Shape inputShape = new Shape(16, 3, 32, 32);

                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                Shape[] outputShape =
                        model.getBlock().getOutputShapes(manager, new Shape[] {inputShape});

                NDArray data = manager.ones(new Shape(16, 3, 32, 32));
                NDArray label = manager.ones(outputShape[0]);
                Batch batch =
                        new Batch(manager, new NDList(data), new NDList(label), Batchifier.STACK);
                trainer.trainBatch(batch);
            }
        }
    }

    private ZooModel<BufferedImage, Classifications> getModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "50");
        criteria.put("dataset", "cifar10");
        return ModelZoo.RESNET.loadModel(criteria);
    }

    private static class TestTranslator implements Translator<NDList, NDList> {

        /** {@inheritDoc} */
        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            return list;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, NDList input) {
            return input;
        }
    }
}
