/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.FruitsFreshAndRotten;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.OneHot;
import ai.djl.modality.cv.transform.RandomFlipLeftRight;
import ai.djl.modality.cv.transform.RandomFlipTopBottom;
import ai.djl.modality.cv.transform.RandomResizedCrop;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.FixedPerVarTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public final class TransferFreshFruitTrainExperiment {

    private TransferFreshFruitTrainExperiment() {}

    public static void main(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        TransferFreshFruitTrainExperiment.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, TranslateException, ModelException, URISyntaxException {
        String[] fruits = {"apple", "banana", "orange"};
        boolean[] trainParams = {true, false};
        TrainingResult result = null;
        for (String fruit : fruits) {
            for (boolean trainParam : trainParams) {
                Criteria<NDList, NDList> criteria =
                        Criteria.builder()
                                .setTypes(NDList.class, NDList.class)
                                .optModelUrls("djl://ai.djl.pytorch/resnet18_embedding")
                                .optEngine("PyTorch")
                                .optProgress(new ProgressBar())
                                .optOption("trainParam", String.valueOf(trainParam))
                                .build();

                ZooModel<NDList, NDList> embedding = criteria.loadModel();

                Block baseBlock = embedding.getBlock();
                Block blocks =
                        new SequentialBlock()
                                .add(baseBlock)
                                .addSingleton(nd -> nd.squeeze(new int[] {2, 3}))
                                .add(Linear.builder().setUnits(2).build()) // linear on which dim?
                                .addSingleton(nd -> nd.softmax(1));

                Model model = Model.newInstance("TransferFreshFruit");
                model.setBlock(blocks);

                try (NDManager manager = NDManager.newBaseManager()) {
                    NDArray cuts = manager.arange(10, 110, 10);
                    NDArray cut2 = manager.create(new int[] {2});
                    cuts = cut2.concat(cuts);
                    float[] accuracy = new float[(int) cuts.size()];

                    int cnt = 0;
                    for (long cut : cuts.toIntArray()) {
                        DefaultTrainingConfig config = setupTrainingConfig(baseBlock);
                        Trainer trainer = model.newTrainer(config);
                        trainer.setMetrics(new Metrics());

                        int batchSize = 32;
                        trainer.initialize(new Shape(batchSize, 3, 224, 224));

                        RandomAccessDataset datasetTrain = getData(fruit, "train", batchSize, cut);
                        RandomAccessDataset datasetTest = getData(fruit, "test", batchSize, 0);
                        EasyTrain.fit(trainer, 49, datasetTrain, null);
                        EasyTrain.fit(trainer, 1, datasetTrain, datasetTest);
                        result = trainer.getTrainingResult();
                        accuracy[cnt++] = result.getValidateEvaluation("Accuracy");
                    }
                    cuts.setName("cuts_" + fruit + "_" + trainParam);
                    NDArray acc = manager.create(accuracy, new Shape(accuracy.length));
                    acc.setName("accuracy_" + fruit + "_" + trainParam);
                    saveNDArray(cuts);
                    saveNDArray(acc);
                }
                model.close();
                embedding.close();
            }
        }
        return result;
    }

    private static void saveNDArray(NDArray array) throws IOException {
        Path path = Paths.get("build").resolve(array.getName() + ".npz");
        try (OutputStream os = Files.newOutputStream(path)) {
            new NDList(new NDList(array)).encode(os, true);
        }
    }

    private static RandomAccessDataset getData(String fruit, String usage, int batchSize, long cut)
            throws TranslateException, IOException {
        // The dataset is accessible from:
        // https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
        float[] mean = {0.485f, 0.456f, 0.406f};
        float[] std = {0.229f, 0.224f, 0.225f};
        String localPath = "localPath";
        Repository repository =
                Repository.newInstance(
                        fruit, Paths.get(localPath + "/" + fruit + "_full/" + usage));
        FruitsFreshAndRotten dataset =
                FruitsFreshAndRotten.builder()
                        .optRepository(repository)
                        .addTransform(new RandomResizedCrop(256, 256)) // only in training
                        .addTransform(new RandomFlipTopBottom()) // only in training
                        .addTransform(new RandomFlipLeftRight()) // only in training
                        .addTransform(new Resize(256, 256))
                        .addTransform(new CenterCrop(224, 224))
                        .addTransform(new ToTensor())
                        .addTransform(new Normalize(mean, std))
                        .addTargetTransform(new OneHot(2))
                        .setSampling(batchSize, true)
                        .build();
        dataset.prepare();

        if (cut > 0) {
            List<Long> batchIndexList = new ArrayList<>();
            try (NDManager manager = NDManager.newBaseManager()) {
                NDArray indices = manager.randomPermutation(dataset.size());
                NDArray batchIndex = indices.get(":{}", cut);
                for (long index : batchIndex.toLongArray()) {
                    batchIndexList.add(index);
                }
            }
            return dataset.subDataset(batchIndexList);
        }
        return dataset;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Block baseBlock) {
        String outputDir = "build/fruits";
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(new SoftmaxCrossEntropy("SoftmaxCrossEntropy"))
                        .addEvaluator(new Accuracy())
                        .optDevices(Engine.getInstance().getDevices(1))
                        .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                        .addTrainingListeners(listener);

        // Customized learning rate
        float lr = 0.0001f;
        FixedPerVarTracker.Builder learningRateTrackerBuilder =
                FixedPerVarTracker.builder().setDefaultValue(lr);
        for (Pair<String, Parameter> paramPair : baseBlock.getParameters()) {
            learningRateTrackerBuilder.put(paramPair.getValue().getId(), 0.1f * lr);
        }
        FixedPerVarTracker learningRateTracker = learningRateTrackerBuilder.build();
        Optimizer optimizer = Adam.builder().optLearningRateTracker(learningRateTracker).build();
        config.optOptimizer(optimizer);

        return config;
    }

    private static class SoftmaxCrossEntropy extends Loss {

        /**
         * Base class for metric with abstract update methods.
         *
         * @param name The display name of the Loss
         */
        public SoftmaxCrossEntropy(String name) {
            super(name);
        }

        /** {@inheritDoc} */
        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            // Here the labels are supposed to be one-hot
            int classAxis = -1;
            NDArray pred = predictions.singletonOrThrow().log();
            NDArray lab = labels.singletonOrThrow().reshape(pred.getShape());
            NDArray loss = pred.mul(lab).neg().sum(new int[] {classAxis}, true);
            return loss.mean();
        }
    }
}
