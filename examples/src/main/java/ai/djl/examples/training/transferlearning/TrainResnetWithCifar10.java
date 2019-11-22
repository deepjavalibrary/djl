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
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.Cifar10;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.metrics.Accuracy;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.optimizer.learningrate.LearningRateTracker;
import ai.djl.training.optimizer.learningrate.MultiFactorTracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.zoo.ModelZoo;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class TrainResnetWithCifar10 extends AbstractTraining {

    public static void main(String[] args) {
        new TrainResnetWithCifar10().runExample(args);
    }

    /** {@inheritDoc} */
    @Override
    protected void train(Arguments arguments) throws IOException, ModelNotFoundException {
        try (Model model = getModel(arguments)) {
            batchSize = arguments.getBatchSize();
            // get training dataset
            Dataset trainDataset = getDataset(model.getNDManager(), Dataset.Usage.TRAIN, arguments);
            Dataset validationDataset =
                    getDataset(model.getNDManager(), Dataset.Usage.TEST, arguments);

            // setup training configuration
            TrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);

                /*
                 * CIFAR10 is 32x32 image and pre processed into NCHW NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, 3, 32, 32);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);
                TrainingUtils.fit(
                        trainer,
                        arguments.getEpoch(),
                        trainDataset,
                        validationDataset,
                        arguments.getOutputDir(),
                        "resnetv1");
            }

            // save model
            model.setProperty("Epoch", String.valueOf(arguments.getEpoch()));
            model.setProperty("Accuracy", String.format("%.2f", getValidationAccuracy()));
            model.save(Paths.get("build/model"), "resnetv1");
        } catch (MalformedModelException e) {
            throw new IllegalArgumentException(e);
        }
    }

    private Model getModel(Arguments arguments)
            throws IOException, ModelNotFoundException, MalformedModelException {
        boolean isSymbolic = arguments.isSymbolic();
        boolean preTrained = arguments.isPreTrained();
        Map<String, String> criteria = arguments.getCriteria();
        if (isSymbolic) {
            // load the model
            if (criteria == null) {
                criteria = new ConcurrentHashMap<>();
                criteria.put("layers", "50");
                criteria.put("flavor", "v1");
            }
            Model model = MxModelZoo.RESNET.loadModel(criteria, new ProgressBar());
            SequentialBlock newBlock = new SequentialBlock();
            SymbolBlock block = (SymbolBlock) model.getBlock();
            block.removeLastBlock();
            newBlock.add(block);
            newBlock.add(x -> new NDList(x.singletonOrThrow().squeeze()));
            newBlock.add(new Linear.Builder().setOutChannels(10).build());
            newBlock.add(Blocks.batchFlattenBlock());
            model.setBlock(newBlock);
            if (!preTrained) {
                model.getBlock().clear();
            }
            return model;
        }
        // imperative resnet50
        if (preTrained) {
            if (criteria == null) {
                criteria = new ConcurrentHashMap<>();
                criteria.put("layers", "50");
                criteria.put("flavor", "v1");
                criteria.put("dataset", "cifar10");
            }
            // load pre-trained imperative ResNet50 from DJL model zoo
            return ModelZoo.RESNET.loadModel(criteria, new ProgressBar());
        } else {
            // construct new ResNet50 without pre-trained weights
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

    private TrainingConfig setupTrainingConfig(Arguments arguments) {
        // epoch number to change learning rate
        int[] epochs;
        if (arguments.isPreTrained()) {
            epochs = new int[] {2, 5, 8};
        } else {
            epochs = new int[] {20, 60, 90, 120, 180};
        }
        int[] steps = Arrays.stream(epochs).map(k -> k * 60000 / batchSize).toArray();
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);
        MultiFactorTracker learningRateTracker =
                LearningRateTracker.multiFactorTracker()
                        .setSteps(steps)
                        .optBaseLearningRate(1e-3f)
                        .optFactor((float) Math.sqrt(.1f))
                        .optWarmUpBeginLearningRate(1e-4f)
                        .optWarmUpSteps(200)
                        .build();
        Optimizer optimizer =
                Optimizer.adam()
                        .setRescaleGrad(1.0f / batchSize)
                        .optLearningRateTracker(learningRateTracker)
                        .optWeightDecays(0.001f)
                        .optClipGrad(5f)
                        .build();
        loss = Loss.softmaxCrossEntropyLoss();
        return new DefaultTrainingConfig(initializer, loss)
                .setOptimizer(optimizer)
                .addTrainingMetric(new Accuracy())
                .setBatchSize(batchSize)
                .setDevices(Device.getDevices(arguments.getMaxGpus()));
    }

    private Dataset getDataset(NDManager manager, Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Pipeline pipeline =
                new Pipeline(
                        new ToTensor(),
                        new Normalize(
                                new float[] {0.4914f, 0.4822f, 0.4465f},
                                new float[] {0.2023f, 0.1994f, 0.2010f}));
        long maxIterations = arguments.getMaxIterations();
        Cifar10 cifar10 =
                Cifar10.builder(manager)
                        .optUsage(usage)
                        .setSampling(batchSize, true)
                        .optMaxIteration(maxIterations)
                        .optPipeline(pipeline)
                        .build();
        cifar10.prepare(new ProgressBar());
        int dataSize = (int) Math.min(cifar10.size() / batchSize, maxIterations);
        if (usage == Dataset.Usage.TRAIN) {
            trainDataSize = dataSize;
        } else if (usage == Dataset.Usage.TEST) {
            validateDataSize = dataSize;
        }
        return cifar10;
    }
}
