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

import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.CaptchaDataset;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.engine.Engine;
import ai.djl.examples.training.util.Arguments;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Dataset.Usage;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SimpleCompositeLoss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import java.io.IOException;

/**
 * An example of training a CAPTCHA solving model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_captcha.md">doc</a>
 * for information about this example.
 */
public final class TrainCaptcha {

    private TrainCaptcha() {}

    public static void main(String[] args) throws IOException, TranslateException {
        TrainCaptcha.runExample(args);
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }

        try (Model model = Model.newInstance("captcha")) {
            model.setBlock(getBlock());

            // get training and validation dataset
            RandomAccessDataset trainingSet = getDataset(Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Usage.VALIDATION, arguments);

            // setup training configuration
            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape =
                        new Shape(1, 1, CaptchaDataset.IMAGE_HEIGHT, CaptchaDataset.IMAGE_WIDTH);

                // initialize trainer with proper input shape
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                return trainer.getTrainingResult();
            }
        }
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("acc_digit_0");
                    model.setProperty("Accuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });
        SimpleCompositeLoss loss = new SimpleCompositeLoss();
        for (int i = 0; i < CaptchaDataset.CAPTCHA_LENGTH; i++) {
            loss.addLoss(new SoftmaxCrossEntropyLoss("loss_digit_" + i), i);
        }

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                        .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                        .addTrainingListeners(listener);

        for (int i = 0; i < CaptchaDataset.CAPTCHA_LENGTH; i++) {
            config.addEvaluator(new Accuracy("acc_digit_" + i, i));
        }

        return config;
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        CaptchaDataset dataset =
                CaptchaDataset.builder()
                        .optUsage(usage)
                        .setSampling(arguments.getBatchSize(), true)
                        .optLimit(arguments.getLimit())
                        .build();
        dataset.prepare(new ProgressBar());
        return dataset;
    }

    private static Block getBlock() {
        Block resnet =
                ResNetV1.builder()
                        .setNumLayers(50)
                        .setImageShape(
                                new Shape(
                                        1, CaptchaDataset.IMAGE_HEIGHT, CaptchaDataset.IMAGE_WIDTH))
                        .setOutSize(CaptchaDataset.CAPTCHA_OPTIONS * CaptchaDataset.CAPTCHA_LENGTH)
                        .build();

        return new SequentialBlock()
                .add(resnet)
                .add(
                        resnetOutputList -> {
                            NDArray resnetOutput = resnetOutputList.singletonOrThrow();
                            NDList splitOutput =
                                    resnetOutput
                                            .reshape(
                                                    -1,
                                                    CaptchaDataset.CAPTCHA_LENGTH,
                                                    CaptchaDataset.CAPTCHA_OPTIONS)
                                            .split(CaptchaDataset.CAPTCHA_LENGTH, 1);

                            NDList output = new NDList(CaptchaDataset.CAPTCHA_LENGTH);
                            for (NDArray outputDigit : splitOutput) {
                                output.add(outputDigit.squeeze(1));
                            }
                            return output;
                        });
    }
}
