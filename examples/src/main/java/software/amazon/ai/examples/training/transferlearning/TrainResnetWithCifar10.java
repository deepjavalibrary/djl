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
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.mxnet.dataset.Cifar10;
import org.apache.mxnet.dataset.SimpleDataset;
import org.apache.mxnet.zoo.ModelNotFoundException;
import org.apache.mxnet.zoo.ModelZoo;
import org.apache.mxnet.zoo.ZooModel;
import org.slf4j.Logger;
import software.amazon.ai.Model;
import software.amazon.ai.examples.inference.util.LogUtils;
import software.amazon.ai.modality.Classification;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.nn.Block;
import software.amazon.ai.nn.SequentialBlock;
import software.amazon.ai.nn.SymbolBlock;
import software.amazon.ai.nn.core.Linear;
import software.amazon.ai.training.Gradient;
import software.amazon.ai.training.Loss;
import software.amazon.ai.training.Trainer;
import software.amazon.ai.training.TrainingController;
import software.amazon.ai.training.dataset.Dataset;
import software.amazon.ai.training.dataset.Record;
import software.amazon.ai.training.initializer.Initializer;
import software.amazon.ai.training.metrics.Accuracy;
import software.amazon.ai.training.metrics.LossMetric;
import software.amazon.ai.training.optimizer.Adam;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.translate.TranslateException;

public final class TrainResnetWithCifar10 {

    private static Logger logger = LogUtils.getLogger(TrainResnetWithCifar10.class);

    private TrainResnetWithCifar10() {}

    public static void main(String[] args)
            throws IOException, ModelNotFoundException, TranslateException {
        // load the model
        Map<String, String> criteria = new ConcurrentHashMap<>();
        criteria.put("layers", "152");
        criteria.put("flavor", "v1d");
        ZooModel<BufferedImage, List<Classification>> model = ModelZoo.RESNET.loadModel(criteria);
        trainCifar10(model);
    }

    public static Block reconstructBlock(Model model) {
        Block modifiedBlock = ((SymbolBlock) model.getBlock()).removeLastBlock();
        SequentialBlock newBlock = new SequentialBlock();
        newBlock.add(modifiedBlock);
        Linear linear = new Linear.Builder().setOutChannels(10).build();
        linear.setInitializer(model.getManager(), Initializer.ONES, true);
        newBlock.add(linear);
        return newBlock;
    }

    public static void trainCifar10(Model model) throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager()) {
            Block reconstructedBlock = reconstructBlock(model);
            int batchSize = 50;
            int numEpoch = 2;
            Cifar10 cifar10 =
                    new Cifar10.Builder()
                            .setManager(manager)
                            .setUsage(Dataset.Usage.TRAIN)
                            .setSampling(batchSize)
                            .build();

            cifar10.prepare();

            Optimizer optimizer = new Adam.Builder().setRescaleGrad(1.0f / batchSize).build();

            TrainingController controller =
                    new TrainingController(reconstructedBlock.getParameters(), optimizer);
            Accuracy acc = new Accuracy();
            LossMetric lossMetric = new LossMetric("softmaxCELoss");

            try (Trainer<NDArray, NDArray, NDArray> trainer =
                    Trainer.newInstance(
                            reconstructedBlock, new SimpleDataset.DefaultTranslator())) {
                for (int epoch = 0; epoch < numEpoch; epoch++) {
                    for (Record batch : trainer.trainDataset(cifar10)) {
                        NDArray data = batch.getData().head().transpose(0, 3, 1, 2).div(255f);
                        NDArray label = batch.getLabels().head();
                        NDArray pred;
                        NDArray loss;
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
                        batch.close();
                        float lossValue = lossMetric.getMetric().getValue();
                        float accuracy = acc.getMetric().getValue();
                        logger.info(
                                "The loss value is " + lossValue + " and accuracy: " + accuracy);
                    }
                    logger.info("Epoch " + epoch + " finish");
                }
            }
        }
    }
}
