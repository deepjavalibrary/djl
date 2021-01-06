/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.training.hyperparameter;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.hyperparameter.optimizer.HpORandom;
import ai.djl.training.hyperparameter.optimizer.HpOptimizer;
import ai.djl.training.hyperparameter.param.HpSet;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Helper for easy training with hyperparameters. */
public abstract class EasyHpo {

    private static final Logger logger = LoggerFactory.getLogger(EasyHpo.class);

    /**
     * Fits the model given the implemented abstract methods.
     *
     * @return the best model and training results
     * @throws IOException for various exceptions depending on the dataset
     * @throws TranslateException if there is an error while processing input
     */
    public Pair<Model, TrainingResult> fit() throws IOException, TranslateException {

        // get training and validation dataset
        RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN);
        RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST);

        HpSet hyperParams = setupHyperParams();
        HpOptimizer hpOptimizer = new HpORandom(hyperParams);

        final int hyperparameterTests = numHyperParameterTests();

        for (int i = 0; i < hyperparameterTests; i++) {
            HpSet hpVals = hpOptimizer.nextConfig();
            Pair<Model, TrainingResult> trained = train(hpVals, trainingSet, validateSet);
            trained.getKey().close();
            float loss = trained.getValue().getValidateLoss();
            hpOptimizer.update(hpVals, loss);
            logger.info(
                    "--------- hp test {}/{} - Loss {} - {}", i, hyperparameterTests, loss, hpVals);
        }

        HpSet bestHpVals = hpOptimizer.getBest().getKey();
        Pair<Model, TrainingResult> trained = train(bestHpVals, trainingSet, validateSet);
        TrainingResult result = trained.getValue();

        Model model = trained.getKey();
        saveModel(model, result);
        return trained;
    }

    private Pair<Model, TrainingResult> train(
            HpSet hpVals, RandomAccessDataset trainingSet, RandomAccessDataset validateSet)
            throws IOException, TranslateException {

        // Construct neural network
        Model model = buildModel(hpVals);

        // setup training configuration
        TrainingConfig config = setupTrainingConfig(hpVals);

        try (Trainer trainer = model.newTrainer(config)) {
            trainer.setMetrics(new Metrics());

            // initialize trainer with proper input shape
            trainer.initialize(inputShape(hpVals));

            EasyTrain.fit(trainer, numEpochs(hpVals), trainingSet, validateSet);

            TrainingResult result = trainer.getTrainingResult();
            return new Pair<>(model, result);
        }
    }

    /**
     * Returns the initial hyperparameters.
     *
     * @return the initial hyperparameters
     */
    protected abstract HpSet setupHyperParams();

    /**
     * Returns the dataset to train with.
     *
     * @param usage the usage of the dataset
     * @return the dataset to train with
     * @throws IOException if the dataset could not be loaded
     */
    protected abstract RandomAccessDataset getDataset(Dataset.Usage usage) throws IOException;

    /**
     * Returns the {@link ai.djl.training.TrainingConfig} to use to train each hyperparameter set.
     *
     * @param hpVals the hyperparameters to train with
     * @return the {@link ai.djl.training.TrainingConfig} to use to train each hyperparameter set
     */
    protected abstract TrainingConfig setupTrainingConfig(HpSet hpVals);

    /**
     * Builds the {@link Model} and {@link ai.djl.nn.Block} to train.
     *
     * @param hpVals the hyperparameter values to use for the model
     * @return the model to train
     */
    protected abstract Model buildModel(HpSet hpVals);

    /**
     * Returns the input shape for the model.
     *
     * @param hpVals the hyperparameter values for the model
     * @return returns the model input shape
     */
    protected abstract Shape inputShape(HpSet hpVals);

    /**
     * Returns the number of epochs to train for the current hyperparameter set.
     *
     * @param hpVals the current hyperparameter set
     * @return the number of epochs
     */
    protected abstract int numEpochs(HpSet hpVals);

    /**
     * Returns the number of hyperparameter sets to train with.
     *
     * @return the number of hyperparameter sets to train with
     */
    protected abstract int numHyperParameterTests();

    /**
     * Saves the best hyperparameter set.
     *
     * @param model the model to save
     * @param result the training result for training with this model's hyperparameters
     * @throws IOException if the model could not be saved
     */
    protected void saveModel(Model model, TrainingResult result) throws IOException {}
}
