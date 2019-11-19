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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicdataset.Mnist;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.FactorTracker;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.zoo.cv.classification.Mlp;
import java.io.IOException;
import java.nio.file.Paths;

public final class TrainMnist extends AbstractTraining {

    public static void main(String[] args) {
        new TrainMnist().runExample(args);
    }

    /** {@inheritDoc} */
    @Override
    protected void train(Arguments arguments) throws IOException {
        // Construct neural network
        Block block = new Mlp(28, 28);

        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            // get training and validation dataset
            Dataset trainingSet = getDataset(model.getNDManager(), Dataset.Usage.TRAIN, arguments);
            Dataset validateSet = getDataset(model.getNDManager(), Dataset.Usage.TEST, arguments);

            // setup training configuration
            TrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);

                /*
                 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, 28 * 28);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainingSet,
                        validateSet,
                        arguments.getOutputDir(),
                        "mlp");
            }

            // save model
            model.setProperty("Epoch", String.valueOf(arguments.getEpoch()));
            model.setProperty("Accuracy", String.format("%.2f", getValidationAccuracy()));
            // TODO: Add more property into model: Throughput, Memory, mAP, Dataset etc.

            model.save(Paths.get(arguments.getOutputDir()), "mlp");
        }
    }

    private TrainingConfig setupTrainingConfig(Arguments arguments) {
        int batchSize = arguments.getBatchSize();
        FactorTracker factorTracker =
                LearningRateTracker.factorTracker()
                        .optBaseLearningRate(0.1f)
                        .setStep(60000 / batchSize)
                        .optFactor(0.1f)
                        .optWarmUpBeginLearningRate(0.01f)
                        .optWarmUpSteps(500)
                        .optStopFactorLearningRate(1e-3f)
                        .build();
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(factorTracker)
                        .optWeightDecays(0.001f)
                        .optMomentum(0.9f)
                        .optClipGrad(1f)
                        .build();

        loss = Loss.softmaxCrossEntropyLoss();
        return new DefaultTrainingConfig(new XavierInitializer(), loss)
                .setOptimizer(optimizer)
                .addTrainingMetric(new Accuracy())
                .setBatchSize(batchSize)
                .setDevices(Device.getDevices(arguments.getMaxGpus()));
    }

    private Dataset getDataset(NDManager manager, Dataset.Usage usage, Arguments arguments)
            throws IOException {
        int batchSize = arguments.getBatchSize();
        long maxIterations = arguments.getMaxIterations();

        Mnist mnist =
                Mnist.builder(manager)
                        .optUsage(usage)
                        .setSampling(batchSize, true)
                        .optMaxIteration(maxIterations)
                        .build();
        mnist.prepare(new ProgressBar());
        if (usage == Dataset.Usage.TRAIN) {
            trainDataSize = (int) Math.min(mnist.size() / batchSize, maxIterations);
        } else {
            validateDataSize = (int) Math.min(mnist.size() / batchSize, maxIterations);
        }
        return mnist;
    }
}
