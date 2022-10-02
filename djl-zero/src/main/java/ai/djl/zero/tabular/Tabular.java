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
import ai.djl.basicdataset.tabular.TabularDataset;
import ai.djl.basicmodelzoo.tabular.TabNet;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.TabNetLoss;
import ai.djl.translate.TranslateException;

import java.io.IOException;

/** Tabular takes a NDList as input and output an NDList (for supervised learning). */
public final class Tabular {

    private Tabular() {}
    /**
     * Trains a TabNet model on a custom dataset.
     *
     * <p>In order to train on a custom dataset, you must create a custom {@link TabularDataset} to
     * load your data.
     *
     * @param dataset the data to train with
     * @param featureSize the size of input features from dataset
     * @param labelSize the size of labels from dataset
     * @return the model as a {@link ZooModel}
     * @throws IOException if the dataset could not be loaded
     * @throws TranslateException if the translator has errors
     */
    public static ZooModel<NDList, NDList> train(
            TabularDataset dataset, int featureSize, int labelSize)
            throws IOException, TranslateException {
        Dataset[] splitDataset = dataset.randomSplit(8, 2);
        Dataset trainDataset = splitDataset[0];
        Dataset validateDataset = splitDataset[1];
        Block tabNet = TabNet.builder().setInputDim(featureSize).setOutDim(labelSize).build();

        Model model = Model.newInstance("tabular");
        model.setBlock(tabNet);

        TrainingConfig trainingConfig =
                new DefaultTrainingConfig(new TabNetLoss())
                        .addTrainingListeners(TrainingListener.Defaults.basic());

        try (Trainer trainer = model.newTrainer(trainingConfig)) {
            trainer.initialize(new Shape(1, featureSize));
            EasyTrain.fit(trainer, 20, trainDataset, validateDataset);
        }

        return new ZooModel<>(model, null);
    }
}
