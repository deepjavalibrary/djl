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
package software.amazon.ai.examples.training;

import ai.djl.mxnet.dataset.DatasetUtils;
import ai.djl.mxnet.dataset.Mnist;
import ai.djl.mxnet.dataset.transform.cv.ToTensor;
import java.io.IOException;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import software.amazon.ai.Device;
import software.amazon.ai.Model;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.examples.training.util.Arguments;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.initializer.XavierInitializer;
import software.amazon.ai.training.loss.Loss;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.learningrate.LearningRateTracker;
import software.amazon.ai.translate.Pipeline;

public final class TrainMnist {

    private static final Logger logger = LogUtils.getLogger(TrainMnist.class);

    private static float accuracy;
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
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
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
                new DefaultTrainingConfig(new XavierInitializer(), optimizer, devices);
        try (Model model = Model.newInstance()) {
            Pipeline pipeline = new Pipeline(new ToTensor());
            model.setBlock(block);

            Mnist mnist =
                    new Mnist.Builder()
                            .setManager(model.getNDManager())
                            .setUsage(Dataset.Usage.TRAIN)
                            .setRandomSampling(batchSize)
                            .optPipeline(pipeline)
                            .build();
            mnist.prepare();
            try (Trainer trainer = model.newTrainer(config)) {
                int numEpoch = arguments.getEpoch();
                int numOfSlices = devices.length;

                Shape inputShape = new Shape(batchSize / numOfSlices, 28 * 28);
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                Accuracy acc = new Accuracy();
                LossMetric lossMetric = new LossMetric("softmaxCELoss");
                for (int epoch = 0; epoch < numEpoch; epoch++) {
                    // reset loss and accuracy
                    acc.reset();
                    lossMetric.reset();
                    for (Batch batch : trainer.iterateDataset(mnist)) {
                        Batch[] split = DatasetUtils.split(batch, devices, false);

                        NDArray[] pred = new NDArray[numOfSlices];
                        NDArray[] loss = new NDArray[numOfSlices];
                        try (GradientCollector gradCol = trainer.newGradientCollector()) {
                            for (int i = 0; i < numOfSlices; i++) {
                                // MNIST only has one input
                                NDArray data = split[i].getData().head();
                                NDArray label = split[i].getLabels().head();

                                data = data.reshape(inputShape);

                                pred[i] = trainer.forward(new NDList(data)).head();
                                loss[i] = Loss.softmaxCrossEntropyLoss().getLoss(label, pred[i]);
                                gradCol.backward(loss[i]);
                            }
                        }
                        trainer.step();
                        for (int i = 0; i < numOfSlices; i++) {
                            NDArray label = split[i].getLabels().head();
                            acc.update(label, pred[i]);
                            lossMetric.update(loss[i]);
                        }
                        batch.close();
                    }
                    lossValue = lossMetric.getMetric().getValue();
                    accuracy = acc.getMetric().getValue();
                    logger.info("Loss: " + lossValue + " accuracy: " + accuracy);
                    logger.info("Epoch " + epoch + " finish");
                }
            }
        }
    }

    public static float getAccuracy() {
        return accuracy;
    }

    public static float getLossValue() {
        return lossValue;
    }

    private static Block constructBlock() {
        return new SequentialBlock()
                .add(new Linear.Builder().setOutChannels(128).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(64).build())
                .add(Activation.reluBlock())
                .add(new Linear.Builder().setOutChannels(10).build());
    }
}
