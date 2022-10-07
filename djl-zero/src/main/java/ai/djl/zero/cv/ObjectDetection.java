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
package ai.djl.zero.cv;

import ai.djl.Model;
import ai.djl.basicdataset.cv.ObjectDetectionDataset;
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.zero.Performance;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** ObjectDetection takes an image and extract one or more main subjects in the image. */
public final class ObjectDetection {
    private ObjectDetection() {}

    /**
     * Trains the recommended object detection model on a custom dataset. Currently, trains a
     * SingleShotDetection Model.
     *
     * <p>In order to train on a custom dataset, you must create a custom {@link
     * ObjectDetectionDataset} to load your data.
     *
     * @param dataset the data to train with
     * @param performance to determine the desired model tradeoffs
     * @return the model as a {@link ZooModel} with the {@link Translator} included
     * @throws IOException if the dataset could not be loaded
     * @throws TranslateException if the translator has errors
     */
    public static ZooModel<Image, DetectedObjects> train(
            ObjectDetectionDataset dataset, Performance performance)
            throws IOException, TranslateException {
        List<String> classes = dataset.getClasses();
        int channels = dataset.getImageChannels();
        int width =
                dataset.getImageWidth()
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "The dataset must have a fixed image width"));
        int height =
                dataset.getImageHeight()
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "The dataset must have a fixed image height"));

        Shape imageShape = new Shape(channels, height, width);

        Dataset[] splitDataset = dataset.randomSplit(8, 2);
        Dataset trainDataset = splitDataset[0];
        Dataset validateDataset = splitDataset[1];

        Block block = getSsdTrainBlock(classes.size());
        Model model = Model.newInstance("ObjectDetection");
        model.setBlock(block);

        TrainingConfig trainingConfig =
                new DefaultTrainingConfig(new SingleShotDetectionLoss())
                        .addEvaluator(new SingleShotDetectionAccuracy("classAccuracy"))
                        .addEvaluator(new BoundingBoxError("boundingBoxError"))
                        .addTrainingListeners(TrainingListener.Defaults.basic());

        try (Trainer trainer = model.newTrainer(trainingConfig)) {
            trainer.initialize(new Shape(1).addAll(imageShape));
            EasyTrain.fit(trainer, 50, trainDataset, validateDataset);
        }

        Translator<Image, DetectedObjects> translator =
                SingleShotDetectionTranslator.builder()
                        .addTransform(new ToTensor())
                        .optSynset(classes)
                        .optThreshold(0.6f)
                        .build();

        return new ZooModel<>(model, translator);
    }

    private static Block getSsdTrainBlock(int numClasses) {
        int[] numFilters = {16, 32, 64};
        SequentialBlock baseBlock = new SequentialBlock();
        for (int numFilter : numFilters) {
            baseBlock.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }

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

        return SingleShotDetection.builder()
                .setNumClasses(numClasses)
                .setNumFeatures(3)
                .optGlobalPool(true)
                .setRatios(ratios)
                .setSizes(sizes)
                .setBaseNetwork(baseBlock)
                .build();
    }
}
