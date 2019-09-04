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

import java.io.IOException;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.dataset.Mnist;
import org.apache.mxnet.dataset.SimpleDataset;
import org.slf4j.Logger;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.examples.training.util.Arguments;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Activation;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.initializer.NormalInitializer;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LrTracker;
import software.amazon.ai.translate.TranslateException;

public final class TrainMnist {

    private static Logger logger = LogUtils.getLogger(TrainMnist.class);
    private static float accuracy;
    private static float lossValue;

    private TrainMnist() {}

    public static void main(String[] args) throws IOException, TranslateException, ParseException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        org.apache.commons.cli.CommandLine cmd = parser.parse(options, args, null, false);
        Arguments arguments = new Arguments(cmd);
        // load the model
        trainMnist(arguments);
    }

    public static Block constructBlock(NDManager manager) {
        SequentialBlock mlp = new SequentialBlock();
        mlp.add(new Linear.Builder().setOutChannels(128).build());
        mlp.add(Activation.reluBlock());
        mlp.add(new Linear.Builder().setOutChannels(64).build());
        mlp.add(Activation.reluBlock());
        mlp.add(new Linear.Builder().setOutChannels(10).build());
        mlp.setInitializer(manager, new NormalInitializer(0.01));
        return mlp;
    }

    public static void trainMnist(Arguments arguments) throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block mlp = constructBlock(manager);
            int batchSize = arguments.getBatchSize();
            int numEpoch = arguments.getEpoch();
            Optimizer optimizer =
                    new Sgd.Builder()
                            .setRescaleGrad(1.0f / batchSize)
                            .setLrTracker(LrTracker.fixedLR(0.1f))
                            .optMomentum(0.9f)
                            .build();
            TrainingController controller = new TrainingController(mlp.getParameters(), optimizer);
            Accuracy acc = new Accuracy();
            LossMetric lossMetric = new LossMetric("softmaxCELoss");

            Mnist mnist =
                    new Mnist.Builder()
                            .setManager(manager)
                            .setUsage(Dataset.Usage.TRAIN)
                            .setSampling(batchSize, true, true)
                            .build();

            mnist.prepare();

            try (Trainer<NDArray, NDArray, NDArray> trainer =
                    Trainer.newInstance(mlp, new SimpleDataset.DefaultTranslator())) {
                for (int epoch = 0; epoch < numEpoch; epoch++) {
                    // reset loss and accuracy
                    acc.reset();
                    lossMetric.reset();
                    NDArray loss;
                    for (Record record : trainer.trainDataset(mnist)) {
                        NDArray data =
                                record.getData().head().reshape(batchSize, 28 * 28).div(255f);
                        NDArray label = record.getLabels().head();
                        NDArray pred;
                        try (Gradient.Collector gradCol = Gradient.newCollector()) {
                            pred = trainer.predict(data);
                            loss =
                                    Loss.softmaxCrossEntropyLoss(
                                            label, pred, 1.f, 0, -1, true, false);
                            gradCol.backward(loss);
                        }
                        controller.step();
                        acc.update(label, pred);
                        lossMetric.update(loss);
                        record.close();
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
}
