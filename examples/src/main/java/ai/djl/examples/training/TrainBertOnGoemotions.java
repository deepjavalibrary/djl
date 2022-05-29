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
package ai.djl.examples.training;

import ai.djl.Model;
import ai.djl.basicdataset.nlp.GoEmotions;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.BertGoemotionsDataset;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.transformer.BertBlock;
import ai.djl.nn.transformer.BertPretrainingBlock;
import ai.djl.nn.transformer.BertPretrainingLoss;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.TruncatedNormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.PolynomialDecayTracker;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.tracker.WarmUpTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;

/** Example that performs Bert pretraining on the GoEmotions dataset. */
public final class TrainBertOnGoemotions {

    private static final int DEFAULT_BATCH_SIZE = 48;
    private static final int DEFAULT_EPOCHS = 4;

    private TrainBertOnGoemotions() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainBertOnGoemotions.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        TrainBertOnGoemotions.BertArguments arguments =
                (TrainBertOnGoemotions.BertArguments)
                        new TrainBertOnGoemotions.BertArguments().parseArgs(args);

        // get training and validation dataset
        GoEmotions trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);

        BertGoemotionsDataset dataset =
                new BertGoemotionsDataset(
                        arguments.getBatchSize(), arguments.getLimit(), trainingSet);
        dataset.prepare();

        // Create model & trainer
        try (Model model = createBertPretrainingModel(dataset.getVocabularySize())) {
            TrainingConfig config = createTrainingConfig(arguments);
            try (Trainer trainer = model.newTrainer(config)) {
                // Initialize training
                Shape inputShape = new Shape(dataset.getMaxSequenceLength(), 512);
                trainer.initialize(inputShape, inputShape, inputShape, inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), dataset, null);
                return trainer.getTrainingResult();
            }
        }
    }

    private static TrainingConfig createTrainingConfig(
            TrainBertOnGoemotions.BertArguments arguments) {

        int warmUpStep = 43410 / arguments.getBatchSize() * arguments.getEpoch();

        Tracker learningRateTracker =
                WarmUpTracker.builder()
                        .optWarmUpBeginValue(0f)
                        .optWarmUpSteps(warmUpStep)
                        .optWarmUpMode(WarmUpTracker.Mode.LINEAR)
                        .setMainTracker(
                                PolynomialDecayTracker.builder()
                                        .setBaseValue(5e-5f)
                                        .setEndLearningRate(5e-5f / 1000)
                                        .setDecaySteps(100000)
                                        .optPower(1f)
                                        .build())
                        .build();
        Optimizer optimizer =
                Adam.builder()
                        .optEpsilon(1e-5f)
                        .optLearningRateTracker(learningRateTracker)
                        .build();
        return new DefaultTrainingConfig(new BertPretrainingLoss())
                .optOptimizer(optimizer)
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging());
    }

    private static Model createBertPretrainingModel(long vocabularySize) {
        Block block =
                new BertPretrainingBlock(
                        BertBlock.builder()
                                .micro()
                                .setTokenDictionarySize(Math.toIntExact(vocabularySize)));
        block.setInitializer(new TruncatedNormalInitializer(0.02f), Parameter.Type.WEIGHT);

        Model model = Model.newInstance("Bert Pretraining");
        model.setBlock(block);
        return model;
    }

    private static GoEmotions getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException, EmbeddingException {
        GoEmotions goemotions =
                GoEmotions.builder()
                        .optUsage(usage)
                        .setSampling(arguments.getBatchSize(), true)
                        .optLimit(arguments.getLimit())
                        .build();
        goemotions.prepare(new ProgressBar());
        return goemotions;
    }

    public static class BertArguments extends Arguments {
        /** {@inheritDoc} */
        @Override
        protected void initialize() {
            super.initialize();
            epoch = DEFAULT_EPOCHS;
            batchSize = DEFAULT_BATCH_SIZE;
        }
    }
}
