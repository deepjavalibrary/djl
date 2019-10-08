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

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.mxnet.dataset.Cifar10;
import org.apache.mxnet.dataset.DatasetUtils;
import org.apache.mxnet.dataset.transform.cv.ToTensor;
import org.apache.mxnet.zoo.ModelZoo;
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
import software.amazon.ai.zoo.cv.classification.ResNetV1;

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
            Block modifiedBlock = ((SymbolBlock) model.getBlock()).removeLastBlock();
            newBlock.add(modifiedBlock);
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
                new Sgd.Builder()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.01f))
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
        TrainingConfig config =
                new DefaultTrainingConfig(new NormalInitializer(0.01), optimizer, devices);

        try (Trainer trainer = model.newTrainer(config)) {
            int numEpoch = arguments.getEpoch();
            int numOfSlices = devices.length;

            Accuracy acc = new Accuracy();
            LossMetric lossMetric = new LossMetric("softmaxCELoss");

            Shape inputShape = new Shape(batchSize, 3, 32, 32);
            trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

            for (int epoch = 0; epoch < numEpoch; epoch++) {
                // reset loss and accuracy
                acc.reset();
                lossMetric.reset();
                for (Batch batch : trainer.iterateDataset(cifar10)) {
                    Batch[] split = DatasetUtils.split(batch, devices, false);

                    NDList pred = new NDList();
                    NDList loss = new NDList();

                    try (GradientCollector gradCol = trainer.newGradientCollector()) {
                        for (int i = 0; i < numOfSlices; i++) {
                            NDArray data = split[i].getData().head();
                            NDArray label = split[i].getLabels().head();
                            NDArray prediction = trainer.forward(new NDList(data)).head();
                            NDArray l =
                                    Loss.softmaxCrossEntropyLoss(
                                            label, prediction, 1.f, 0, -1, true, false);
                            pred.add(prediction);
                            loss.add(l);
                            gradCol.backward(l);
                        }
                    }
                    trainer.step();
                    acc.update(
                            new NDList(
                                    Arrays.stream(split)
                                            .map(Batch::getLabels)
                                            .map(NDList::head)
                                            .toArray(NDArray[]::new)),
                            pred);
                    lossMetric.update(loss);
                    lossValue = lossMetric.getMetric().getValue();
                    accuracy = acc.getMetric().getValue();
                    logger.info("Loss: " + lossValue + " accuracy: " + accuracy);
                    for (Batch b : split) {
                        b.close();
                    }
                    pred.close();
                    loss.close();
                    batch.close();
                }
                lossValue = lossMetric.getMetric().getValue();
                accuracy = acc.getMetric().getValue();
                logger.info("Loss: " + lossValue + " accuracy: " + accuracy);
                logger.info("Epoch " + epoch + " finish");
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
