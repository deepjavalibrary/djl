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

package ai.djl.examples.training;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.engine.Engine;
import ai.djl.examples.inference.DeepARTimeSeries;
import ai.djl.examples.training.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.repository.Repository;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.dataset.M5Forecast;
import ai.djl.timeseries.dataset.TimeFeaturizers;
import ai.djl.timeseries.distribution.DistributionLoss;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.evaluator.Rmsse;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.timefeature.TimeFeature;
import ai.djl.timeseries.transform.TimeSeriesTransform;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Progress;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** An example of training a deepar timeseries model. */
public final class TrainTimeSeries {

    private static final Logger logger = LoggerFactory.getLogger(TrainTimeSeries.class);
    private static String freq = "W";
    private static int predictionLength = 4;
    private static LocalDateTime startTime = LocalDateTime.parse("2011-01-29T00:00");

    private TrainTimeSeries() {}

    public static void main(String[] args) throws IOException, TranslateException, ModelException {
        TrainTimeSeries.runExample(args);
        Map<String, Float> metrics = predict("build/model");
        for (Map.Entry<String, Float> entry : metrics.entrySet()) {
            logger.info(String.format("metric: %s:\t%.2f", entry.getKey(), entry.getValue()));
        }
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        // use data path to create a custom repository
        Repository repository =
                Repository.newInstance(
                        "test",
                        Paths.get(
                                System.getProperty("user.home")
                                        + "/Desktop/m5-forecasting-accuracy"));

        Arguments arguments = new Arguments().parseArgs(args);
        try (Model model = Model.newInstance("deepar")) {
            // specify the model distribution output, for M5 case, NegativeBinomial best describe it
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DefaultTrainingConfig config = setupTrainingConfig(arguments, distributionOutput);

            NDManager manager = model.getNDManager();
            DeepARNetwork trainingNetwork = getDeepARModel(distributionOutput, true);
            model.setBlock(trainingNetwork);

            List<TimeSeriesTransform> trainingTransformation =
                    trainingNetwork.createTrainingTransformation(manager);
            int contextLength = trainingNetwork.getContextLength();

            M5Forecast trainSet =
                    getDataset(
                            trainingTransformation, repository, contextLength, Dataset.Usage.TRAIN);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                int historyLength = trainingNetwork.getHistoryLength();
                Shape[] inputShapes = new Shape[9];
                // (N, num_cardinality)
                inputShapes[0] = new Shape(1, 5);
                // (N, num_real) if use_feat_stat_real else (N, 1)
                inputShapes[1] = new Shape(1, 1);
                // (N, history_length, num_time_feat + num_age_feat)
                inputShapes[2] =
                        new Shape(
                                1,
                                historyLength,
                                TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
                inputShapes[3] = new Shape(1, historyLength);
                inputShapes[4] = new Shape(1, historyLength);
                inputShapes[5] = new Shape(1, historyLength);
                inputShapes[6] =
                        new Shape(
                                1,
                                predictionLength,
                                TimeFeature.timeFeaturesFromFreqStr(freq).size() + 1);
                inputShapes[7] = new Shape(1, predictionLength);
                inputShapes[8] = new Shape(1, predictionLength);
                trainer.initialize(inputShapes);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainSet, null);
                return trainer.getTrainingResult();
            }
        }
    }

    public static Map<String, Float> predict(String outputDir)
            throws IOException, TranslateException, ModelException {
        Repository repository =
                Repository.newInstance(
                        "test",
                        Paths.get(
                                System.getProperty("user.home")
                                        + "/Desktop/m5-forecasting-accuracy"));

        try (Model model = Model.newInstance("deepar")) {
            DeepARNetwork predictionNetwork = getDeepARModel(new NegativeBinomialOutput(), false);
            model.setBlock(predictionNetwork);
            model.load(Paths.get(outputDir));

            M5Forecast testSet =
                    getDataset(
                            new ArrayList<>(),
                            repository,
                            predictionNetwork.getContextLength(),
                            Dataset.Usage.TEST);

            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("prediction_length", predictionLength);
            arguments.put("freq", freq);
            arguments.put("use_" + FieldName.FEAT_DYNAMIC_REAL.name().toLowerCase(), false);
            arguments.put("use_" + FieldName.FEAT_STATIC_CAT.name().toLowerCase(), true);
            arguments.put("use_" + FieldName.FEAT_STATIC_REAL.name().toLowerCase(), false);
            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();

            DeepARTimeSeries.M5Evaluator evaluator =
                    new DeepARTimeSeries.M5Evaluator(0.5f, 0.67f, 0.95f, 0.99f);
            Progress progress = new ProgressBar();
            progress.reset("Inferring", testSet.size());
            try (Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor(translator)) {
                for (Batch batch : testSet.getData(model.getNDManager().newSubManager())) {
                    NDList data = batch.getData();
                    NDArray target = data.head();
                    NDArray featStaticCat = data.get(1);

                    NDArray gt = target.get(":, {}:", -predictionLength);
                    NDArray pastTarget = target.get(":, :{}", -predictionLength);

                    NDList gtSplit = gt.split(batch.getSize());
                    NDList pastTargetSplit = pastTarget.split(batch.getSize());
                    NDList featStaticCatSplit = featStaticCat.split(batch.getSize());

                    List<TimeSeriesData> batchInput = new ArrayList<>(batch.getSize());
                    for (int i = 0; i < batch.getSize(); i++) {
                        TimeSeriesData input = new TimeSeriesData(10);
                        input.setStartTime(startTime);
                        input.setField(FieldName.TARGET, pastTargetSplit.get(i).squeeze(0));
                        input.setField(
                                FieldName.FEAT_STATIC_CAT, featStaticCatSplit.get(i).squeeze(0));
                        batchInput.add(input);
                    }
                    List<Forecast> forecasts = predictor.batchPredict(batchInput);
                    for (int i = 0; i < forecasts.size(); i++) {
                        evaluator.aggregateMetrics(
                                evaluator.getMetricsPerTs(
                                        gtSplit.get(i).squeeze(0),
                                        pastTargetSplit.get(i).squeeze(0),
                                        forecasts.get(i)));
                    }
                    progress.increment(batch.getSize());
                    batch.close();
                }
                return evaluator.computeTotalMetrics();
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(
            Arguments arguments, DistributionOutput distributionOutput) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float rmsse = result.getValidateEvaluation("RMSSE");
                    model.setProperty("RMSSE", String.format("%.5f", rmsse));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new DistributionLoss("Loss", distributionOutput))
                .addEvaluator(new Rmsse(distributionOutput))
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    /**
     * Create the deepar model with specified distribution output.
     *
     * @param distributionOutput the distribution output
     * @param training if training create trainingNetwork else predictionNetwork
     * @return deepar model
     */
    private static DeepARNetwork getDeepARModel(
            DistributionOutput distributionOutput, boolean training) {
        // here is feat_static_cat's cardinality which depend on your dataset
        List<Integer> cardinality = new ArrayList<>();
        cardinality.add(3);
        cardinality.add(10);
        cardinality.add(3);
        cardinality.add(7);
        cardinality.add(3049);

        DeepARNetwork.Builder builder =
                DeepARNetwork.builder()
                        .setCardinality(cardinality)
                        .setFreq(freq)
                        .setPredictionLength(predictionLength)
                        .optDistrOutput(distributionOutput)
                        .optUseFeatStaticCat(true);
        return training ? builder.buildTrainingNetwork() : builder.buildPredictionNetwork();
    }

    private static M5Forecast getDataset(
            List<TimeSeriesTransform> transformation,
            Repository repository,
            int contextLength,
            Dataset.Usage usage)
            throws IOException {
        // In order to create a TimeSeriesDataset, you must specify the transformation of the data
        // preprocessing
        M5Forecast.Builder builder =
                M5Forecast.builder()
                        .optUsage(usage)
                        .setRepository(repository)
                        .setTransformation(transformation)
                        .setContextLength(contextLength)
                        .setSampling(32, usage == Dataset.Usage.TRAIN);

        int maxWeek = usage == Dataset.Usage.TRAIN ? 273 : 277;
        for (int i = 1; i <= maxWeek; i++) {
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
                                        TimeFeaturizers.getConstantTimeFeaturizer(startTime)))
                        .build();
        m5Forecast.prepare(new ProgressBar());
        return m5Forecast;
    }
}
