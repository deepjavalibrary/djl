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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.basicdataset.cv.PikachuDetection;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.inference.Predictor;
import ai.djl.integration.util.TestUtils;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class SingleShotDetectionTest {

    @Test
    public void testLoadPredict()
            throws IOException, ModelNotFoundException, TranslateException,
                    MalformedModelException {
        try (ZooModel<Image, DetectedObjects> model = getModel()) {
            model.setBlock(getPredictBlock(model.getBlock()));
            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
                Path imagePath = Paths.get("../examples/src/test/resources/pikachu.jpg");
                Image image = ImageFactory.getInstance().fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                int numPikachus = detectedObjects.getNumberOfObjects();
                Assert.assertTrue(numPikachus >= 6);
                Assert.assertTrue(numPikachus <= 15);
            }
        }
    }

    @Test
    public void testLoadTrain()
            throws IOException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        try (ZooModel<Image, DetectedObjects> model = getModel()) {
            TrainingConfig config = setupTrainingConfig();
            try (Trainer trainer = model.newTrainer(config)) {
                Dataset dataset = getDataset();

                Shape inputShape = new Shape(32, 3, 256, 256);
                trainer.initialize(inputShape);
                Iterable<Batch> iterator = dataset.getData(model.getNDManager());
                EasyTrain.trainBatch(trainer, iterator.iterator().next());
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
                                    MultiBoxDetection.builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));
                            return detections.singletonOrThrow().split(new long[] {1, 2}, 2);
                        }));
        return ssdPredict;
    }

    private Dataset getDataset() {
        Pipeline pipeline = new Pipeline(new ToTensor());
        return PikachuDetection.builder()
                .optUsage(Dataset.Usage.TEST)
                .optPipeline(pipeline)
                .setSampling(32, true)
                .optLimit(64)
                .build();
    }

    private TrainingConfig setupTrainingConfig() {
        return new DefaultTrainingConfig(new SingleShotDetectionLoss())
                .addEvaluator(new SingleShotDetectionAccuracy("classAccuracy"))
                .addEvaluator(new BoundingBoxError("boundingBoxError"))
                .optDevices(Device.getDevices(1));
    }

    private ZooModel<Image, DetectedObjects> getModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        if (!TestUtils.isMxnet()) {
            throw new SkipException("SSD-pikachu model only available in MXNet");
        }

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optGroupId(BasicModelZoo.GROUP_ID)
                        .optArtifactId("ssd")
                        .optFilter("flavor", "tiny")
                        .optFilter("dataset", "pikachu")
                        .build();

        return criteria.loadModel();
    }
}
