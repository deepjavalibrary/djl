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
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DataManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.util.ProgressBar;
import java.io.IOException;
import java.nio.file.Paths;
import org.apache.commons.cli.ParseException;

public final class TrainMnistWithLSTM {

    private TrainMnistWithLSTM() {}

    public static void main(String[] args) throws IOException, ParseException {
        TrainMnistWithLSTM.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, ParseException {
        Arguments arguments = Arguments.parseArgs(args);

        try (Model model = Model.newInstance()) {
            model.setBlock(getLSTMModel());

            // get training and validation dataset
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);
            config.addTrainingListeners(
                    TrainingListener.Defaults.logging(arguments.getOutputDir()));

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * MNIST is 28x28 grayscale image and pre processed into 28 * 28 NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(32, 28, 28);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainingSet,
                        validateSet,
                        arguments.getOutputDir(),
                        "lstm");

                TrainingResult result = trainer.getTrainingResult();
                float accuracy = result.getValidateEvaluation("Accuracy");
                model.setProperty("Epoch", String.valueOf(result.getEpoch()));
                model.setProperty("Accuracy", String.format("%.5f", accuracy));
                model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));

                model.save(Paths.get(arguments.getOutputDir()), "lstm");
                return result;
            }
        }
    }

    private static Block getLSTMModel() {
        SequentialBlock block = new SequentialBlock();
        block.add(
                new LSTM.Builder().setStateSize(64).setNumStackedLayers(1).optDropRate(0).build());
        block.add(BatchNorm.builder().optEpsilon(1e-5f).optMomentum(0.9f).build());
        block.add(Linear.builder().setOutChannels(10).optFlatten(true).build());
        return block;
    }

    public static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optInitializer(new XavierInitializer())
                .optDataManager(new MnistWithLSTMDataManager())
                .optDevices(Device.getDevices(arguments.getMaxGpus()));
    }

    public static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Mnist mnist =
                Mnist.builder()
                        .optUsage(usage)
                        .setSampling(arguments.getBatchSize(), false, true)
                        .optLimit(arguments.getLimit())
                        .build();
        mnist.prepare(new ProgressBar());
        return mnist;
    }

    private static class MnistWithLSTMDataManager extends DataManager {
        @Override
        public NDList getData(Batch batch) {
            NDList ret = new NDList();
            NDList data = batch.getData();
            Shape inputShape = data.singletonOrThrow().getShape();
            long batchSize = inputShape.get(0);
            long channel = inputShape.get(3);
            long time = inputShape.size() / (batchSize * channel);
            ret.add(data.singletonOrThrow().reshape(new Shape(batchSize, time, channel)));
            return ret;
        }
    }
}
