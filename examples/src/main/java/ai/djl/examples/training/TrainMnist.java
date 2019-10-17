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
import ai.djl.examples.inference.util.LogUtils;
import ai.djl.examples.training.util.Arguments;
import ai.djl.mxnet.dataset.Mnist;
import ai.djl.mxnet.dataset.transform.cv.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.Activation;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.FactorTracker;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Pipeline;
import java.io.IOException;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;

public final class TrainMnist {

    private static final Logger logger = LogUtils.getLogger(TrainMnist.class);

    private static float trainAccuracy;
    private static float lossValue;

    private TrainMnist() {}

    public static void main(String[] args) throws IOException, ParseException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        org.apache.commons.cli.CommandLine cmd = parser.parse(options, args, null, false);
        Arguments arguments = new Arguments(cmd);
        // load the model
        trainMnist(arguments);
    }

    public static void trainMnist(Arguments arguments) throws IOException {
        int batchSize = arguments.getBatchSize();
        int numGpus = arguments.getNumGpus();
        Block block = constructBlock();
        FactorTracker factorTracker =
                LearningRateTracker.factorTracker()
                        .optBaseLearningRate(0.01f)
                        .setStep(1000)
                        .optFactor(0.1f)
                        .optWarmUpBeginLearningRate(0.001f)
                        .optWarmUpSteps(200)
                        .optStopFactorLearningRate(1e-4f)
                        .build();
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(factorTracker)
                        .optWeightDecays(0.001f)
                        .optMomentum(0.9f)
                        .build();

        Device[] devices;
        if (numGpus > 1) {
            devices = new Device[numGpus];
            for (int i = 0; i < numGpus; i++) {
                devices[i] = Device.gpu(i);
            }
        } else {
            devices = new Device[] {Device.defaultDevice()};
        }

        TrainingConfig config =
                new DefaultTrainingConfig(new XavierInitializer())
                        .setOptimizer(optimizer)
                        .setLoss(Loss.softmaxCrossEntropyLoss())
                        .addTrainingMetrics(new Accuracy())
                        .setDevices(devices);
        try (Model model = Model.newInstance()) {
            model.setBlock(block);

            Pipeline pipeline = new Pipeline(new ToTensor());

            Mnist mnist =
                    new Mnist.Builder()
                            .setManager(model.getNDManager())
                            .setUsage(Dataset.Usage.TRAIN)
                            .setRandomSampling(batchSize)
                            .optPipeline(pipeline)
                            .build();
            mnist.prepare();

            Mnist validateSet =
                    new Mnist.Builder()
                            .setManager(model.getNDManager())
                            .setUsage(Dataset.Usage.TEST)
                            .setRandomSampling(batchSize)
                            .optPipeline(pipeline)
                            .build();
            validateSet.prepare();

            try (Trainer trainer = model.newTrainer(config)) {
                int numEpoch = arguments.getEpoch();
                int numOfSlices = devices.length;

                Shape inputShape = new Shape(batchSize / numOfSlices, 28 * 28);
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                for (int epoch = 0; epoch < numEpoch; epoch++) {
                    for (Batch batch : trainer.iterateDataset(mnist)) {
                        Batch[] split = batch.split(devices, false);

                        try (GradientCollector gradCol = trainer.newGradientCollector()) {
                            for (int i = 0; i < numOfSlices; i++) {
                                // MNIST only has one input
                                NDArray data = split[i].getData().head();
                                NDList labels = split[i].getLabels();

                                NDList preds = trainer.forward(new NDList(data));
                                NDArray loss = trainer.loss(labels, preds);
                                gradCol.backward(loss);
                            }
                        }
                        trainer.step();

                        batch.close();
                    }
                    // Validation
                    for (Batch batch : trainer.iterateDataset(validateSet)) {
                        Batch[] split = batch.split(devices, false);
                        for (int i = 0; i < numOfSlices; i++) {
                            NDArray data = split[i].getData().head();
                            NDList labels = split[i].getLabels();
                            trainer.validate(new NDList(data), labels);
                        }
                        batch.close();
                    }
                    lossValue = trainer.getLoss();
                    float validationLoss = trainer.getValidationLoss();
                    trainAccuracy = trainer.getTrainingMetrics().get(0).getMetric().getValue();
                    float validateAccuracy =
                            trainer.getValidateMetrics().get(0).getMetric().getValue();
                    logger.info(
                            "train loss: "
                                    + lossValue
                                    + " validate loss: "
                                    + validationLoss
                                    + " train accuracy: "
                                    + trainAccuracy
                                    + " validate accuracy: "
                                    + validateAccuracy);
                    logger.info("Epoch " + epoch + " finish");
                    // reset loss and accuracy
                    trainer.resetTrainingMetrics();
                }
            }
        }
    }

    public static float getTrainAccuracy() {
        return trainAccuracy;
    }

    public static float getLossValue() {
        return lossValue;
    }

    private static Block constructBlock() {
        return new SequentialBlock()
                .add(Blocks.flattenBlock(28 * 28))
                .add(new Linear.Builder().setOutChannels(128).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(64).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(10).build());
    }
}
