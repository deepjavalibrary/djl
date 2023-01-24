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
package ai.djl.zero.tabular;

import ai.djl.Model;
import ai.djl.basicdataset.tabular.ListFeatures;
import ai.djl.basicdataset.tabular.TabularDataset;
import ai.djl.basicmodelzoo.tabular.TabNet;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.TabNetRegressionLoss;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.zero.Performance;

import java.io.IOException;

/** TabularRegression takes a NDList as input and output an NDList (for supervised learning). */
public final class TabularRegression {

    private TabularRegression() {}

    /**
     * Trains a Model on a custom dataset. Currently, trains a TabNet Model.
     *
     * <p>In order to train on a custom dataset, you must create a custom {@link TabularDataset} to
     * load your data.
     *
     * @param dataset the data to train with
     * @param performance to determine the desired model tradeoffs
     * @return the model as a {@link ZooModel}
     * @throws IOException if the dataset could not be loaded
     * @throws TranslateException if the translator has errors
     */
    public static ZooModel<ListFeatures, Float> train(
            TabularDataset dataset, Performance performance)
            throws IOException, TranslateException {
        Dataset[] splitDataset = dataset.randomSplit(8, 2);
        Dataset trainDataset = splitDataset[0];
        Dataset validateDataset = splitDataset[1];
        int featureSize = dataset.getFeatureSize();
        int labelSize = dataset.getLabelSize();

        Block block;
        if (performance.equals(Performance.FAST)) {
            // for fast cases, we set the number of independent layers and shared layers lower
            block =
                    TabNet.builder()
                            .setInputDim(featureSize)
                            .setOutDim(labelSize)
                            .optNumIndependent(1)
                            .optNumShared(1)
                            .build();
        } else if (performance.equals(Performance.BALANCED)) {
            block = TabNet.builder().setInputDim(featureSize).setOutDim(labelSize).build();
        } else {
            // for accurate cases, we set the number of independent layers and shared layers higher
            block =
                    TabNet.builder()
                            .setInputDim(featureSize)
                            .setOutDim(labelSize)
                            .optNumIndependent(4)
                            .optNumShared(4)
                            .build();
        }

        Model model = Model.newInstance("tabular");
        model.setBlock(block);

        TrainingConfig trainingConfig =
                new DefaultTrainingConfig(new TabNetRegressionLoss())
                        .addTrainingListeners(TrainingListener.Defaults.basic());

        try (Trainer trainer = model.newTrainer(trainingConfig)) {
            trainer.initialize(new Shape(1, featureSize));
            EasyTrain.fit(trainer, 20, trainDataset, validateDataset);
        }

        Translator<ListFeatures, Float> translator =
                dataset.matchingTranslatorOptions().option(ListFeatures.class, Float.class);
        return new ZooModel<>(model, translator);
    }
}
