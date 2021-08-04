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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.Cifar10;
import ai.djl.basicmodelzoo.BasicModelZoo;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Map;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

/** This example features sample usage of a variety of optimizers to train Cifar10. */
public final class TrainWithOptimizers {

    private TrainWithOptimizers() {}

    public static void main(String[] args)
            throws IOException, ParseException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        TrainWithOptimizers.runExample(args);
    }

    public static TrainingResult runExample(String[] args)
            throws IOException, ParseException, ModelNotFoundException, MalformedModelException,
                    TranslateException {
        OptimizerArguments arguments =
                (OptimizerArguments) new OptimizerArguments().parseArgs(args);

        try (Model model = getModel(arguments)) {
            // get training dataset
            RandomAccessDataset trainDataset = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validationDataset = getDataset(Dataset.Usage.TEST, arguments);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                /*
                 * CIFAR10 is 32x32 image and pre processed into NCHW NDArray.
                 * 1st axis is batch axis, we can use 1 for initialization.
                 */
                Shape inputShape = new Shape(1, 3, Cifar10.IMAGE_HEIGHT, Cifar10.IMAGE_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);
                EasyTrain.fit(trainer, arguments.getEpoch(), trainDataset, validationDataset);

                return trainer.getTrainingResult();
            }
        }
    }

    private static Model getModel(Arguments arguments)
            throws IOException, ModelNotFoundException, MalformedModelException {
        boolean isSymbolic = arguments.isSymbolic();
        boolean preTrained = arguments.isPreTrained();
        Map<String, String> options = arguments.getCriteria();
        Criteria.Builder<Image, Classifications> builder =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optProgress(new ProgressBar())
                        .optArtifactId("resnet");
        if (isSymbolic) {
            // currently only MxEngine support removeLastBlock
            builder.optGroupId("ai.djl.mxnet");
            if (options == null) {
                builder.optFilter("layers", "50");
                builder.optFilter("flavor", "v1");
            } else {
                builder.optFilters(options);
            }

            Model model = builder.build().loadModel();
            SequentialBlock newBlock = new SequentialBlock();
            SymbolBlock block = (SymbolBlock) model.getBlock();
            block.removeLastBlock();
            newBlock.add(block);
            // the original model don't include the flatten
            // so apply the flatten here
            newBlock.add(Blocks.batchFlattenBlock());
            newBlock.add(Linear.builder().setUnits(10).build());
            model.setBlock(newBlock);
            if (!preTrained) {
                model.getBlock().clear();
            }
            return model;
        }
        // imperative resnet50
        if (preTrained) {
            builder.optGroupId(BasicModelZoo.GROUP_ID);
            if (options == null) {
                builder.optFilter("layers", "50");
                builder.optFilter("flavor", "v1");
                builder.optFilter("dataset", "cifar10");
            } else {
                builder.optFilters(options);
            }
            // load pre-trained imperative ResNet50 from DJL model zoo
            return builder.build().loadModel();
        } else {
            // construct new ResNet50 without pre-trained weights
            Model model = Model.newInstance("resnetv1");
            Block resNet50 =
                    ResNetV1.builder()
                            .setImageShape(new Shape(3, Cifar10.IMAGE_HEIGHT, Cifar10.IMAGE_WIDTH))
                            .setNumLayers(50)
                            .setOutSize(10)
                            .build();
            model.setBlock(resNet50);
            return model;
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(OptimizerArguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir, "resnetv1");
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("Accuracy");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                .addEvaluator(new Accuracy())
                .optOptimizer(setupOptimizer(arguments))
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    private static Optimizer setupOptimizer(OptimizerArguments arguments) {
        String optimizerName = arguments.getOptimizer();
        int batchSize = arguments.getBatchSize();
        switch (optimizerName) {
            case "sgd":
                // epoch number to change learning rate
                int[] epochs;
                if (arguments.isPreTrained()) {
                    epochs = new int[] {2, 5, 8};
                } else {
                    epochs = new int[] {20, 60, 90, 120, 180};
                }
                int[] steps = Arrays.stream(epochs).map(k -> k * 60000 / batchSize).toArray();
                Tracker learningRateTracker =
                        Tracker.warmUp()
                                .optWarmUpBeginValue(1e-4f)
                                .optWarmUpSteps(200)
                                .setMainTracker(
                                        Tracker.multiFactor()
                                                .setSteps(steps)
                                                .setBaseValue(1e-3f)
                                                .optFactor((float) Math.sqrt(.1f))
                                                .build())
                                .build();
                return Optimizer.sgd()
                        .setLearningRateTracker(learningRateTracker)
                        .optWeightDecays(0.001f)
                        .optClipGrad(5f)
                        .build();
            case "adam":
                return Optimizer.adam().build();
            default:
                throw new IllegalArgumentException("Unknown optimizer");
        }
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Pipeline pipeline =
                new Pipeline(
                        new ToTensor(),
                        new Normalize(Cifar10.NORMALIZE_MEAN, Cifar10.NORMALIZE_STD));
        Cifar10 cifar10 =
                Cifar10.builder()
                        .optUsage(usage)
                        .setSampling(arguments.getBatchSize(), true)
                        .optLimit(arguments.getLimit())
                        .optPipeline(pipeline)
                        .build();
        cifar10.prepare(new ProgressBar());
        return cifar10;
    }

    private static class OptimizerArguments extends Arguments {

        private String optimizer;

        public OptimizerArguments() {}

        @Override
        protected void setCmd(CommandLine cmd) {
            super.setCmd(cmd);
            if (cmd.hasOption("optimizer")) {
                optimizer = cmd.getOptionValue("optimizer");
            } else {
                optimizer = "adam";
            }
        }

        @Override
        public Options getOptions() {
            Options options = super.getOptions();
            options.addOption(
                    Option.builder("z")
                            .longOpt("optimizer")
                            .hasArg()
                            .argName("OPTIMIZER")
                            .desc("The optimizer to use.")
                            .build());
            return options;
        }

        public String getOptimizer() {
            return optimizer;
        }
    }
}
