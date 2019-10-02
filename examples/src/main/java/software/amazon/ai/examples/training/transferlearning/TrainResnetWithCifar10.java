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
package software.amazon.ai.examples.training.transferlearning;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.dataset.Cifar10;
import org.apache.mxnet.dataset.transform.cv.ToTensor;
import org.apache.mxnet.zoo.ModelZoo;
import org.slf4j.Logger;
import software.amazon.ai.Model;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.nn.SymbolBlock;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.DefaultTrainingConfig;
import software.amazon.ai.training.GradientCollector;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingConfig;
import software.amazon.ai.training.dataset.Batch;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.initializer.NormalInitializer;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.training.optimizer.Sgd;
import software.amazon.ai.training.optimizer.learningrate.LearningRateTracker;
import software.amazon.ai.translate.Pipeline;
import software.amazon.ai.zoo.ModelNotFoundException;
import software.amazon.ai.zoo.ZooModel;

public final class TrainResnetWithCifar10 {

    private static final Logger logger = LogUtils.getLogger(TrainResnetWithCifar10.class);

    private TrainResnetWithCifar10() {}

    public static void main(String[] args) throws IOException, ModelNotFoundException {
        // load the model
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "152");
        criteria.put("flavor", "v1d");
        ZooModel<BufferedImage, Classification> model = ModelZoo.RESNET.loadModel(criteria);
        trainCifar10(model);
        model.close();
    }

    public static void trainCifar10(Model model) throws IOException {
        reconstructBlock(model);

        int batchSize = 50;
        int numEpoch = 2;
        Optimizer optimizer =
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.1f))
                        .build();
        Pipeline pipeline = new Pipeline(new ToTensor());
        Cifar10 cifar10 =
                new Cifar10.Builder()
                        .setManager(model.getNDManager())
                        .setUsage(Dataset.Usage.TRAIN)
                        .setRandomSampling(batchSize)
                        .optPipeline(pipeline)
                        .build();
        cifar10.prepare();

        TrainingConfig config =
                new DefaultTrainingConfig(new NormalInitializer(0.01), false, optimizer);

        try (Trainer trainer = model.newTrainer(config)) {
            Accuracy acc = new Accuracy();
            LossMetric lossMetric = new LossMetric("softmaxCELoss");

            for (int epoch = 0; epoch < numEpoch; epoch++) {
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    NDList data = batch.getData();
                    NDArray label = batch.getLabels().head();
                    NDArray pred;
                    NDArray loss;
                    try (GradientCollector gradCol = trainer.newGradientCollector()) {
                        pred = trainer.forward(data).get(0);
                        loss = Loss.softmaxCrossEntropyLoss(label, pred, 1.f, 0, -1, true, false);
                        gradCol.backward(loss);
                    }
                    trainer.step();
                    acc.update(label, pred);
                    lossMetric.update(loss);
                    batch.close();
                    float lossValue = lossMetric.getMetric().getValue();
                    float accuracy = acc.getMetric().getValue();
                    logger.info("The loss value is " + lossValue + " and accuracy: " + accuracy);
                }
                logger.info("Epoch " + epoch + " finish");
            }
        }
    }

    private static void reconstructBlock(Model model) {
        SequentialBlock newBlock = new SequentialBlock();
        Block modifiedBlock = ((SymbolBlock) model.getBlock()).removeLastBlock();
        newBlock.add(modifiedBlock);
        newBlock.add(new Linear.Builder().setOutChannels(10).build());
        model.setBlock(newBlock);
    }
}
