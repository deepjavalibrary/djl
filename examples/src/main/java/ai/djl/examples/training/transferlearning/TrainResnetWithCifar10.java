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

import ai.djl.Model;
import ai.djl.basicdataset.Cifar10;
import ai.djl.examples.training.util.AbstractTraining;
import ai.djl.examples.training.util.Arguments;
import ai.djl.examples.training.util.TrainingUtils;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataDesc;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
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
import ai.djl.translate.Pipeline;
import ai.djl.zoo.cv.classification.ResNetV1;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class TrainResnetWithCifar10 extends AbstractTraining {

    public static void main(String[] args) {
        new TrainResnetWithCifar10().runExample(args);
    }

    @Override
    protected void train(Arguments arguments) throws IOException, ModelNotFoundException {
        // setup training configuration
        TrainingConfig config = setupTrainingConfig(arguments);

        // configure input data shape based on batch size
        int batchSize = arguments.getBatchSize();
        Shape inputShape = new Shape(batchSize, 3, 32, 32);

        try (Model model = getModel(arguments.getIsSymbolic(), arguments.getPreTrained())) {

            // get training dataset
            Dataset dataset = getDataset(model.getNDManager(), arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(metrics);
                trainer.setTrainingListener(this);

                // initialize trainer
                trainer.initialize(new DataDesc[] {new DataDesc(inputShape)});

                TrainingUtils.fit(trainer, config, dataset, null);
            }

            // save model
            if (arguments.getOutputDir() != null) {
                model.save(Paths.get(arguments.getOutputDir()), "resnet");
            }
        }
    }

    private Model getModel(boolean isSymbolic, boolean preTrained)
            throws IOException, ModelNotFoundException {
        if (isSymbolic) {
            // load the model
            Map<String, String> criteria = new ConcurrentHashMap<>();
            criteria.put("layers", "152");
            criteria.put("flavor", "v1d");
            Model model = MxModelZoo.RESNET.loadModel(criteria);
            SequentialBlock newBlock = new SequentialBlock();
            SymbolBlock block = (SymbolBlock) model.getBlock();
            block.removeLastBlock();
            newBlock.add(block);
            newBlock.add(x -> new NDList(x.singletonOrThrow().squeeze()));
            newBlock.add(new Linear.Builder().setOutChannels(10).build());
            model.setBlock(newBlock);
            if (!preTrained) {
                model.getBlock().clear();
            }
            return model;
        }

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

    private TrainingConfig setupTrainingConfig(Arguments arguments) {
        int batchSize = arguments.getBatchSize();
        Initializer initializer =
                new XavierInitializer(
                        XavierInitializer.RandomType.UNIFORM, XavierInitializer.FactorType.AVG, 2);
        MultiFactorTracker learningRateTracker =
                LearningRateTracker.multiFactorTracker()
                        .setSteps(new int[] {1000, 3000, 8000})
                        .optBaseLearningRate(0.01f)
                        .optFactor(0.1f)
                        .optWarmUpBeginLearningRate(1e-3f)
                        .optWarmUpSteps(300)
                        .build();
        Optimizer optimizer =
                Optimizer.sgd()
                        .setRescaleGrad(1.0f / batchSize)
                        .setLearningRateTracker(learningRateTracker)
                        .optMomentum(0.9f)
                        .optWeightDecays(0.001f)
                        .optClipGrad(1f)
                        .build();

        return new DefaultTrainingConfig(initializer)
                .setOptimizer(optimizer)
                .addTrainingMetric(Loss.softmaxCrossEntropyLoss())
                .addTrainingMetric(new Accuracy())
                .setEpoch(arguments.getEpoch())
                .setBatchSize(arguments.getBatchSize());
    }

    private Dataset getDataset(NDManager manager, Arguments arguments) throws IOException {
        Pipeline pipeline = new Pipeline(new ToTensor());
        int batchSize = arguments.getBatchSize();
        long maxIterations = arguments.getMaxIterations();

        Cifar10 cifar10 =
                new Cifar10.Builder()
                        .setManager(manager)
                        .setUsage(Dataset.Usage.TRAIN)
                        .setRandomSampling(batchSize)
                        .optMaxIteration(maxIterations)
                        .optPipeline(pipeline)
                        .build();
        cifar10.prepare();
        trainDataSize = (int) Math.min(cifar10.size() / batchSize, maxIterations);
        return cifar10;
    }
}
