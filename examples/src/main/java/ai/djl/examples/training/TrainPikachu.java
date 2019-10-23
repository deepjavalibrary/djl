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
import ai.djl.basicdataset.PikachuDetection;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.modality.cv.MultiBoxTarget;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.BatchSampler;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomSampler;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.SSDMultiBoxLoss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.metrics.SsdBoxPredictionError;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.Sgd;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.zoo.cv.object_detection.ssd.SingleShotDetection;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class TrainPikachu extends AbstractTraining {

    public static void main(String[] args) {
        new TrainPikachu().runExample(args);
    }

    @Override
    protected void train(Arguments arguments) throws IOException {
        int batchSize = arguments.getBatchSize();
        MultiBoxTarget multiBoxTarget = new MultiBoxTarget.Builder().build();
        List<List<Float>> sizes = new ArrayList<>();
        List<List<Float>> ratios = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ratios.add(Arrays.asList(1f, 2f, 0.5f));
        }
        sizes.add(Arrays.asList(0.2f, 0.272f));
        sizes.add(Arrays.asList(0.37f, 0.447f));
        sizes.add(Arrays.asList(0.54f, 0.619f));
        sizes.add(Arrays.asList(0.71f, 0.79f));
        sizes.add(Arrays.asList(0.88f, 0.961f));
        Optimizer optimizer =
                new Sgd.Builder()
                        .setLearningRateTracker(LearningRateTracker.fixedLearningRate(0.2f))
                        .setRescaleGrad(1.0f / batchSize)
                        .optWeightDecays(5e-4f)
                        .build();
        SSDMultiBoxLoss ssdLoss = new SSDMultiBoxLoss("ssd loss");
        TrainingConfig config =
                new DefaultTrainingConfig(new XavierInitializer(), ssdLoss).setOptimizer(optimizer);
        Device[] devices = config.getDevices();
        try (Model model = Model.newInstance()) {
            Block network = getBaseNetwork();
            SingleShotDetection block =
                    new SingleShotDetection.Builder()
                            .setManager(model.getNDManager())
                            .setNumClasses(1)
                            .setNumFeatures(3)
                            .optGlobalPool(true)
                            .setRatios(ratios)
                            .setSizes(sizes)
                            .optNetwork(network)
                            .build();
            model.setBlock(block);

            PikachuDetection pikachuDetectionTrainingSet =
                    new PikachuDetection.Builder()
                            .optUsage(Dataset.Usage.TRAIN)
                            .setSampler(new BatchSampler(new RandomSampler(), batchSize, false))
                            .build();
            pikachuDetectionTrainingSet.prepare();

            try (Trainer trainer = model.newTrainer(config)) {
                int numEpoch = arguments.getEpoch();
                int numOfSlices = devices.length;

                Shape inputShape = new Shape(batchSize / numOfSlices, 3, 256, 256);
                trainer.initialize(inputShape);

                Accuracy classAccuracy = new Accuracy("classAccuracy", 0, -1);
                SsdBoxPredictionError boxAccuracy =
                        new SsdBoxPredictionError("SSDBoxPredictionError");
                for (int epoch = 0; epoch < numEpoch; epoch++) {
                    for (Batch batch : trainer.iterateDataset(pikachuDetectionTrainingSet)) {
                        Batch[] split = batch.split(devices, false);

                        NDList[] pred = new NDList[numOfSlices];
                        NDArray[] loss = new NDArray[numOfSlices];
                        NDArray[] bboxLabels = new NDArray[numOfSlices];
                        NDArray[] bboxMasks = new NDArray[numOfSlices];
                        NDArray[] classLabels = new NDArray[numOfSlices];
                        try (GradientCollector gradCol = trainer.newGradientCollector()) {
                            for (int i = 0; i < numOfSlices; i++) {
                                // Pikachu dataset has only has one input
                                NDArray data = split[i].getData().head();
                                NDArray label = split[i].getLabels().head();
                                pred[i] = trainer.forward(new NDList(data));
                                NDArray classPred = pred[i].get(0);
                                NDList target =
                                        multiBoxTarget.target(
                                                new NDList(
                                                        block.getAnchorBoxes(),
                                                        label,
                                                        classPred.transpose(0, 2, 1)));
                                loss[i] = ssdLoss.getLoss(target, pred[i]);
                                bboxLabels[i] = target.get(0);
                                bboxMasks[i] = target.get(1);
                                classLabels[i] = target.get(2);
                                gradCol.backward(loss[i]);
                            }
                        }
                        trainer.step();
                        float classEval = 0;
                        float boxEval = 0;
                        for (int i = 0; i < numOfSlices; i++) {
                            classAccuracy.update(classLabels[i], pred[i].get(0));
                            boxAccuracy.update(
                                    new NDList(bboxLabels[i], bboxMasks[i]),
                                    new NDList(pred[i].get(1)));
                            classEval =
                                    classEval
                                            + pred[i].get(0)
                                                    .argmax(-1)
                                                    .eq(classLabels[i])
                                                    .sum()
                                                    .toFloatArray()[0];
                            boxEval =
                                    boxEval
                                            + bboxLabels[i]
                                                    .sub(pred[i].get(1))
                                                    .mul(bboxMasks[i])
                                                    .abs()
                                                    .sum()
                                                    .toFloatArray()[0];
                        }
                        batch.close();
                    }
                    trainer.resetTrainingMetrics();
                    // reset loss and accuracy
                    classAccuracy.reset();
                    boxAccuracy.reset();
                }
            }
            model.save(Paths.get("/Users/kvasist/models"), "SSD-2");
        }
    }

    public static Block getBaseNetwork() {
        int[] numFilters = {16, 32, 64};
        SequentialBlock block = new SequentialBlock();
        for (int numFilter : numFilters) {
            block.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }
        return block;
    }
}
