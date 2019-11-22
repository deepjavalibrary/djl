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

package ai.djl.integration.tests.model_zoo.object_detection;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.basicdataset.PikachuDetection;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.metrics.BoundingBoxError;
import ai.djl.training.metrics.SingleShotDetectionAccuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import ai.djl.zoo.ModelZoo;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SingleShotDetectionTest {
    @Test
    public void testLoadPredict()
            throws IOException, ModelNotFoundException, TranslateException,
                    MalformedModelException {
        try (ZooModel<BufferedImage, DetectedObjects> model = getModel()) {
            model.setBlock(getPredictBlock(model.getBlock()));
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                Path imagePath = Paths.get("../examples/src/test/resources/pikachu.jpg");
                BufferedImage image = BufferedImageUtils.fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                int numPikachus = detectedObjects.getNumberOfObjects();
                Assert.assertTrue(numPikachus >= 6);
                Assert.assertTrue(numPikachus <= 15);
            }
        }
    }

    @Test
    public void testLoadTrain()
            throws IOException, ModelNotFoundException, MalformedModelException {
        try (ZooModel<BufferedImage, DetectedObjects> model = getModel()) {
            TrainingConfig config = setupTrainingConfig();
            try (Trainer trainer = model.newTrainer(config)) {
                Dataset dataset = getDataset();

                Shape inputShape = new Shape(32, 3, 256, 256);
                trainer.initialize(inputShape);
                Iterable<Batch> iterator = dataset.getData(model.getNDManager());
                trainer.trainBatch(iterator.iterator().next());
            }
        }
    }

    private Block getPredictBlock(Block trainBlock) {
        SequentialBlock ssdPredict = new SequentialBlock();
        ssdPredict.add(trainBlock);
        ssdPredict.add(
                new LambdaBlock(
                        output -> {
                            NDArray anchors = output.get(0);
                            NDArray classPredictions = output.get(1).softmax(-1).transpose(0, 2, 1);
                            NDArray boundingBoxPredictions = output.get(2);
                            MultiBoxDetection multiBoxDetection =
                                    new MultiBoxDetection.Builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));
                            return detections.singletonOrThrow().split(new int[] {1, 2}, 2);
                        }));
        return ssdPredict;
    }

    private Dataset getDataset() throws IOException {
        Pipeline pipeline = new Pipeline(new ToTensor());
        PikachuDetection pikachuDetection =
                new PikachuDetection.Builder()
                        .optUsage(Dataset.Usage.TEST)
                        .optPipeline(pipeline)
                        .setSampling(32, true)
                        .build();
        pikachuDetection.prepare(new ProgressBar());
        return pikachuDetection;
    }

    private TrainingConfig setupTrainingConfig() {
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / 32)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.2f))
                        .optWeightDecays(5e-4f)
                        .build();
        return new DefaultTrainingConfig(initializer, new SingleShotDetectionLoss("ssd_loss"))
                .setOptimizer(optimizer)
                .setBatchSize(32)
                .addTrainingMetric(new SingleShotDetectionAccuracy("classAccuracy"))
                .addTrainingMetric(new BoundingBoxError("boundingBoxError"))
                .setDevices(Device.getDevices(1));
    }

    private ZooModel<BufferedImage, DetectedObjects> getModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("flavor", "tiny");
        criteria.put("dataset", "pikachu");
        return ModelZoo.SSD.loadModel(criteria);
    }
}
