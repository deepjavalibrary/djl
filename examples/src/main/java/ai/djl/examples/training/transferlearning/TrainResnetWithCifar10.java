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
package ai.djl.examples.training.transferlearning;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.examples.inference.util.LogUtils;
import ai.djl.examples.training.util.Arguments;
import ai.djl.mxnet.dataset.Cifar10;
import ai.djl.mxnet.dataset.DatasetUtils;
import ai.djl.mxnet.dataset.transform.cv.ToTensor;
import ai.djl.mxnet.zoo.ModelZoo;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
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
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.translate.Pipeline;
import ai.djl.zoo.ModelNotFoundException;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;

public final class TrainResnetWithCifar10 {

    private static final Logger logger = LogUtils.getLogger(TrainResnetWithCifar10.class);

    private static float accuracy;
    private static float lossValue;

    private TrainResnetWithCifar10() {}

    public static void main(String[] args)
            throws IOException, ParseException, ModelNotFoundException {
        Options options = Arguments.getOptions();
        DefaultParser parser = new DefaultParser();
        org.apache.commons.cli.CommandLine cmd = parser.parse(options, args, null, false);
        Arguments arguments = new Arguments(cmd);
        // load the model
        Model resnet50v1 = getModel(arguments.getIsSymbolic(), arguments.getPreTrained());
        trainResNetV1(resnet50v1, arguments);
        resnet50v1.close();
    }

    private static Model getModel(boolean isSymbolic, boolean preTrained)
            throws IOException, ModelNotFoundException {
        if (isSymbolic) {
            // load the model
            Map<String, String> criteria = new ConcurrentHashMap<>();
            criteria.put("layers", "152");
            criteria.put("flavor", "v1d");
            Model model = ModelZoo.RESNET.loadModel(criteria);
            SequentialBlock newBlock = new SequentialBlock();
            SymbolBlock block = (SymbolBlock) model.getBlock();
            block.removeLastBlock();
            newBlock.add(block);
            newBlock.add(x -> new NDList(x.head().squeeze()));
            newBlock.add(new Linear.Builder().setOutChannels(10).build());
            model.setBlock(newBlock);
            if (!preTrained) {
                model.getBlock().clear();
            }
            return model;
        } else {
            Model model = Model.newInstance();
            Block resNet50 =
                    new ResNetV1.Builder()
                            .setImageShape(new Shape(3, 32, 32))
                            .setNumLayers(50)
                            .setOutSize(10)
                            .build();
            model.setBlock(resNet50);
            return model;
        }
    }

    public static void trainResNetV1(Model model, Arguments arguments) throws IOException {
        int batchSize = arguments.getBatchSize();
        int numGpus = arguments.getNumGpus();

        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.01f))
                        .optClipGrad(1f)
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

        Device[] devices;
        if (numGpus > 0) {
            devices = new Device[numGpus];
            for (int i = 0; i < numGpus; i++) {
                devices[i] = Device.gpu(i);
            }
        } else {
            devices = new Device[] {Device.defaultDevice()};
        }
        Accuracy acc = new Accuracy();

        TrainingConfig config =
                new DefaultTrainingConfig(new XavierInitializer())
                        .setOptimizer(optimizer)
                        .setDevices(devices)
                        .setLoss(Loss.softmaxCrossEntropyLoss())
                        .addTrainingMetrics(Collections.singletonList(acc));

        try (Trainer trainer = model.newTrainer(config)) {
            int numEpoch = arguments.getEpoch();
            int numOfSlices = devices.length;

            Shape inputShape = new Shape(batchSize, 3, 32, 32);
            trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

            for (int epoch = 0; epoch < numEpoch; epoch++) {
                // reset loss and accuracy
                trainer.resetTrainingMetrics();
                int batchNum = 0;
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    batchNum++;
                    Batch[] split = DatasetUtils.split(batch, devices, false);

                    try (GradientCollector gradCol = trainer.newGradientCollector()) {
                        for (int i = 0; i < numOfSlices; i++) {
                            NDList data = split[i].getData();
                            NDList label = split[i].getLabels();
                            NDList pred = trainer.forward(data);
                            NDArray l = trainer.loss(label, pred);
                            gradCol.backward(l);
                        }
                    }
                    trainer.step();
                    lossValue = trainer.getLoss();
                    accuracy = acc.getMetric().getValue();
                    logger.info(
                            "[Epoch "
                                    + epoch
                                    + ", Batch "
                                    + batchNum
                                    + "/"
                                    + (cifar10.size() / batchSize)
                                    + "] - Loss: "
                                    + lossValue
                                    + " accuracy: "
                                    + accuracy);
                    for (Batch b : split) {
                        b.close();
                    }
                    batch.close();
                }
                lossValue = trainer.getLoss();
                accuracy = trainer.getTrainingMetrics().get(0).getMetric().getValue();
                logger.info("Loss: " + lossValue + " accuracy: " + accuracy);
                logger.info("Epoch " + epoch + " finish");
            }

            if (arguments.getOutputDir() != null) {
                model.save(Paths.get(arguments.getOutputDir()), "resnet");
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
