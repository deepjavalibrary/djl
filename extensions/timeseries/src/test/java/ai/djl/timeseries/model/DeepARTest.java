/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.model;

import ai.djl.Model;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.testing.TestRequirements;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.distribution.output.StudentTOutput;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class DeepARTest {

    private static int predictionLength = 4;
    private static String freq = "D";

    @BeforeClass
    public void setUp() {
        // TODO: Remove this once we support PyTorch support for timeseries extension
        TestRequirements.notArm();
    }

    @Test
    public void testTrainingNetwork() {
        try (Model model = Model.newInstance("deepar")) {
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            TrainingConfig config =
                    new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                            .optDevices(Engine.getInstance().getDevices());

            NDManager manager = model.getNDManager();
            DeepARNetwork deepARTraining = getDeepARModel(distributionOutput, true);
            model.setBlock(deepARTraining);
            try (Trainer trainer = model.newTrainer(config)) {
                int batchSize = 1;
                int historyLength = deepARTraining.getHistoryLength();
                Shape[] inputShapes = getTrainingInputShapes(batchSize, historyLength);
                trainer.initialize(inputShapes);

                NDList inputs =
                        new NDList(
                                Stream.of(inputShapes)
                                        .map(manager::ones)
                                        .collect(Collectors.toList()));
                NDArray label =
                        manager.ones(
                                new Shape(
                                        batchSize,
                                        deepARTraining.getContextLength() + predictionLength - 1));
                Batch batch =
                        new Batch(
                                manager.newSubManager(),
                                inputs,
                                new NDList(label),
                                batchSize,
                                Batchifier.STACK,
                                Batchifier.STACK,
                                0,
                                0);
                PairList<String, Parameter> parameters = deepARTraining.getParameters();
                EasyTrain.trainBatch(trainer, batch);
                trainer.step();

                Assert.assertEquals(
                        parameters.get(0).getValue().getArray().getShape(), new Shape(1, 40));
                Assert.assertEquals(
                        parameters.get(1).getValue().getArray().getShape(), new Shape(1));
                Assert.assertEquals(
                        parameters.get(2).getValue().getArray().getShape(), new Shape(1, 40));
                Assert.assertEquals(
                        parameters.get(3).getValue().getArray().getShape(), new Shape(1));
                Assert.assertEquals(
                        parameters.get(4).getValue().getArray().getShape(), new Shape(5, 3));
            }
        }
    }

    @Test
    public void testPredictionNetwork() {
        DeepARNetwork deepAR = getDeepARModel(new NegativeBinomialOutput(), false);
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 1;
            int historyLength = deepAR.getHistoryLength();
            Shape[] inputShapes = getPredictionInputShapes(batchSize, historyLength);

            deepAR.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            deepAR.initialize(manager, DataType.FLOAT32, inputShapes);

            ParameterStore ps = new ParameterStore(manager, true);
            NDArray actOutput =
                    deepAR.forward(
                                    ps,
                                    new NDList(
                                            Stream.of(inputShapes)
                                                    .map(manager::ones)
                                                    .collect(Collectors.toList())),
                                    false)
                            .singletonOrThrow();
            NDArray expOutput = manager.ones(new Shape(batchSize, 100, predictionLength));
            Assert.assertEquals(actOutput.getShape(), expOutput.getShape());
        }
    }

    @Test
    public void testOutputShapes() {
        DeepARNetwork deepARTraining = getDeepARModel(new NegativeBinomialOutput(), true);
        DeepARNetwork deepARPrediction = getDeepARModel(new StudentTOutput(), false);
        try (NDManager manager = NDManager.newBaseManager()) {
            int batchSize = 1;
            int historyLength = deepARTraining.getHistoryLength();
            Shape[] trainingInputShapes = getTrainingInputShapes(batchSize, historyLength);
            Shape[] predictionInputShapes = getPredictionInputShapes(batchSize, historyLength);

            deepARTraining.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            deepARTraining.initialize(manager, DataType.FLOAT32, trainingInputShapes);

            deepARPrediction.setInitializer(Initializer.ONES, Parameter.Type.WEIGHT);
            deepARPrediction.initialize(manager, DataType.FLOAT32, predictionInputShapes);

            Shape[] trainingOutputShapes = deepARTraining.getOutputShapes(trainingInputShapes);
            Shape[] predictionOutputShapes =
                    deepARPrediction.getOutputShapes(predictionInputShapes);

            int contextLength = deepARTraining.getContextLength();
            // Distribution param shape (batch_size, context_length - 1 + prediction_length,
            // arg_shape)
            Assert.assertEquals(
                    trainingOutputShapes[0],
                    new Shape(batchSize, contextLength - 1 + predictionLength));
            Assert.assertEquals(
                    trainingOutputShapes[1],
                    new Shape(batchSize, contextLength - 1 + predictionLength));
            // scale shape
            Assert.assertEquals(trainingOutputShapes[2], new Shape(batchSize, 1));
            // loss weights shape
            Assert.assertEquals(
                    trainingOutputShapes[3],
                    new Shape(batchSize, contextLength - 1 + predictionLength));

            // prediction sample shape (batch_size, prediction_length - 1)
            Assert.assertEquals(
                    predictionOutputShapes[0], new Shape(batchSize, 100, predictionLength));
        }
    }

    @Test
    public void testTrainingTransformation() throws IOException, TranslateException {
        DeepARNetwork deepAR = getDeepARModel(new NegativeBinomialOutput(), true);
        try (NDManager manager = NDManager.newBaseManager()) {
            List<TimeSeriesTransform> trainingTransformation =
                    deepAR.createTrainingTransformation(manager);

            int batchSize = 32;
            M5Forecast m5Forecast = getDataset(batchSize, trainingTransformation);
            Batch batch = m5Forecast.getData(manager).iterator().next();
            Assert.assertEquals(batch.getData().size(), 9);
            Assert.assertEquals(batch.getLabels().size(), 1);

            Shape[] actInputShapes =
                    batch.getData().stream().map(NDArray::getShape).toArray(Shape[]::new);
            Assert.assertEquals(
                    actInputShapes, getTrainingInputShapes(batchSize, deepAR.getHistoryLength()));
        }
    }

    @Test
    public void testPredictionTransformation() throws IOException, TranslateException {
        DeepARNetwork deepAR = getDeepARModel(new StudentTOutput(), false);
        try (NDManager manager = NDManager.newBaseManager()) {
            List<TimeSeriesTransform> predictionTransformation =
                    deepAR.createPredictionTransformation(manager);

            int batchSize = 32;
            M5Forecast m5Forecast = getDataset(batchSize, predictionTransformation);
            Batch batch = m5Forecast.getData(manager).iterator().next();
            Assert.assertEquals(batch.getData().size(), 7);
            Assert.assertEquals(batch.getLabels().size(), 0);

            Shape[] actInputShapes =
                    batch.getData().stream().map(NDArray::getShape).toArray(Shape[]::new);
            Assert.assertEquals(
                    actInputShapes, getPredictionInputShapes(batchSize, deepAR.getHistoryLength()));
        }
    }

    private DeepARNetwork getDeepARModel(DistributionOutput distributionOutput, boolean isTrain) {
        // here is feat_static_cat's cardinality which depend on your dataset
        List<Integer> cardinality = Arrays.asList(5);

        DeepARNetwork.Builder builder =
                DeepARNetwork.builder()
                        .setCardinality(cardinality)
                        .setFreq(freq)
                        .setPredictionLength(predictionLength)
                        .optDistrOutput(distributionOutput);
        return isTrain ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
    }

    private Shape[] getTrainingInputShapes(int batchSize, int historyLength) {
        return new Shape[] {
            new Shape(batchSize, 1),
            new Shape(batchSize, 1),
            new Shape(
                    batchSize, historyLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1),
            new Shape(batchSize, historyLength),
            new Shape(batchSize, historyLength),
            new Shape(batchSize, historyLength),
            new Shape(
                    batchSize,
                    predictionLength,
                    TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1),
            new Shape(batchSize, predictionLength),
            new Shape(batchSize, predictionLength),
        };
    }

    private Shape[] getPredictionInputShapes(int batchSize, int historyLength) {
        return new Shape[] {
            new Shape(batchSize, 1),
            new Shape(batchSize, 1),
            new Shape(
                    batchSize, historyLength, TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1),
            new Shape(batchSize, historyLength),
            new Shape(batchSize, historyLength),
            new Shape(
                    batchSize,
                    predictionLength,
                    TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1),
            new Shape(batchSize, historyLength)
        };
    }

    private M5Forecast getDataset(int batchSize, List<TimeSeriesTransform> transforms)
            throws TranslateException, IOException {
        M5Forecast.Builder builder =
                M5Forecast.builder()
                        .optUsage(Dataset.Usage.TEST)
                        .optUsage(Dataset.Usage.TEST)
                        .optRepository(BasicDatasets.REPOSITORY)
                        .optGroupId(BasicDatasets.GROUP_ID)
                        .optArtifactId("m5forecast-unittest")
                        .setTransformation(transforms)
                        .setContextLength(predictionLength)
                        .setSampling(batchSize, true);
        List<String> features = builder.getAvailableFeatures();
        Assert.assertEquals(features.size(), 5);
        for (int i = 1; i <= 277; i++) {
            builder.addFeature("w_" + i, FieldName.TARGET);
        }
        M5Forecast m5Forecast =
                builder.addFeature("state_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("store_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("cat_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("dept_id", FieldName.FEAT_STATIC_CAT)
                        .addFeature("item_id", FieldName.FEAT_STATIC_CAT)
                        .addFieldFeature(
                                FieldName.START,
                                new Feature(
                                        "date",
                                        TimeFeaturizers.getConstantTimeFeaturizer(
                                                LocalDateTime.parse("2011-01-29T00:00"))))
                        .build();

        m5Forecast.prepare();
        return m5Forecast;
    }
}
