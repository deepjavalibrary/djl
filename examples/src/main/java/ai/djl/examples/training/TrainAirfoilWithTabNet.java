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
import ai.djl.basicdataset.tabular.AirfoilRandomAccess;
import ai.djl.basicdataset.tabular.ListFeatures;
import ai.djl.basicdataset.tabular.TabularDataset;
import ai.djl.basicdataset.tabular.TabularResults;
import ai.djl.basicmodelzoo.tabular.TabNet;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.TabNetRegressionLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;

import java.io.IOException;

public final class TrainAirfoilWithTabNet {
    private TrainAirfoilWithTabNet() {}

    public static void main(String[] args) throws TranslateException, IOException {
        TrainAirfoilWithTabNet.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        // Construct a tabNet instance
        Block tabNet = TabNet.builder().setInputDim(5).setOutDim(1).build();

        try (Model model = Model.newInstance("tabNet")) {
            model.setBlock(tabNet);

            // get the training and validation dataset
            TabularDataset dataset = getDataset(arguments);
            RandomAccessDataset[] split = dataset.randomSplit(8, 2);
            RandomAccessDataset trainingSet = split[0];
            RandomAccessDataset validateSet = split[1];

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(arguments.getBatchSize(), 5);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                Translator<ListFeatures, TabularResults> translator =
                        dataset.matchingTranslatorOptions()
                                .option(ListFeatures.class, TabularResults.class);
                try (Predictor<ListFeatures, TabularResults> predictor =
                        model.newPredictor(translator)) {
                    ListFeatures input =
                            new ListFeatures(dataset.getRowDirect(3, dataset.getFeatures()));
                    predictor.predict(input);
                }

                return trainer.getTrainingResult();
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new TabNetRegressionLoss())
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static TabularDataset getDataset(Arguments arguments)
            throws IOException, TranslateException {
        AirfoilRandomAccess.Builder airfoilBuilder = AirfoilRandomAccess.builder();

        // only train dataset is available, so we get train dataset and split them
        airfoilBuilder.optUsage(Dataset.Usage.TRAIN).setSampling(arguments.getBatchSize(), true);

        for (int i = 0; i < airfoilBuilder.getAvailableFeatures().size() - 1; i++) {
            airfoilBuilder.addFeature(airfoilBuilder.getAvailableFeatures().get(i));
        }

        AirfoilRandomAccess airfoilRandomAccess = airfoilBuilder.build();
        airfoilRandomAccess.prepare(new ProgressBar());
        // split the dataset into
        return airfoilRandomAccess;
    }
}
