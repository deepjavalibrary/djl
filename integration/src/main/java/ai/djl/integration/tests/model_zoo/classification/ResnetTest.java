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

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.integration.util.TestUtils;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.testing.Assertions;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.loss.Loss;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import java.io.IOException;
import java.util.Collections;
import java.util.List;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class ResnetTest {

    @Test
    public void testTrain() {
        TrainingConfig config =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .optDevices(Engine.getInstance().getDevices(2))
                        .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);

        Block resNet50 =
                ResNetV1.builder()
                        .setImageShape(new Shape(1, 28, 28))
                        .setNumLayers(50)
                        .setOutSize(10)
                        .build();
        try (Model model = Model.newInstance("resnet")) {
            model.setBlock(resNet50);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                Shape inputShape = new Shape(batchSize, 1, 28, 28);
                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();

                NDArray input = manager.ones(inputShape);
                NDArray label = manager.ones(new Shape(batchSize, 1));
                Batch batch =
                        new Batch(
                                manager.newSubManager(),
                                new NDList(input),
                                new NDList(label),
                                batchSize,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                0);
                PairList<String, Parameter> parameters = resNet50.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();
                NDArray expectedAtIndex0 = manager.ones(new Shape(16, 1, 3, 3));
                NDArray expectedAtIndex1 = manager.ones(new Shape(16, 16, 3, 3));
                NDArray expectedAtIndex87 = manager.ones(new Shape(32));
                Assertions.assertAlmostEquals(
                        parameters.get(0).getValue().getArray(), expectedAtIndex0);
                Assertions.assertAlmostEquals(
                        parameters.get(1).getValue().getArray(), expectedAtIndex1);
                Assertions.assertAlmostEquals(
                        parameters.get(87).getValue().getArray(), expectedAtIndex87);
            }
        }
    }

    @Test
    public void testLoadPredict()
            throws IOException, ModelNotFoundException, TranslateException,
                    MalformedModelException {
        try (ZooModel<Image, Classifications> model = getModel()) {
            NoopTranslator translator = new NoopTranslator(Batchifier.STACK);
            try (Predictor<NDList, NDList> predictor = model.newPredictor(translator)) {
                NDList input = new NDList(model.getNDManager().ones(new Shape(3, 32, 32)));
                List<NDList> inputs = Collections.nCopies(16, input);
                predictor.batchPredict(inputs);
            }
        }
    }

    @Test
    public void testLoadTrain()
            throws IOException, ModelNotFoundException, MalformedModelException {
        try (ZooModel<Image, Classifications> model = getModel()) {
            TrainingConfig config =
                    new DefaultTrainingConfig(Loss.l1Loss())
                            .optDevices(Engine.getInstance().getDevices(2))
                            .optInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 2;
                Shape inputShape = new Shape(batchSize, 3, 32, 32);

                trainer.initialize(inputShape);

                NDManager manager = trainer.getManager();
                Shape[] outputShape = model.getBlock().getOutputShapes(new Shape[] {inputShape});

                NDArray data = manager.ones(new Shape(batchSize, 3, 32, 32));
                NDArray label = manager.ones(outputShape[0]);
                Batch batch =
                        new Batch(
                                manager.newSubManager(),
                                new NDList(data),
                                new NDList(label),
                                batchSize,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                0);
                EasyTrain.trainBatch(trainer, batch);
            }
        }
    }

    private ZooModel<Image, Classifications> getModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        if (!TestUtils.isMxnet()) {
            throw new SkipException("Resnet50-cifar10 model only available in MXNet");
        }

        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.IMAGE_CLASSIFICATION)
                        .setTypes(Image.class, Classifications.class)
                        .optGroupId(BasicModelZoo.GROUP_ID)
                        .optArtifactId("resnet")
                        .optFilter("layers", "50")
                        .optFilter("dataset", "cifar10")
                        .build();

        return criteria.loadModel();
    }
}
