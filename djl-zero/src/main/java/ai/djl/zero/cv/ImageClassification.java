/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Application.CV;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicdataset.cv.classification.ImageClassificationDataset;
import ai.djl.basicdataset.cv.classification.ImageNet;
import ai.djl.basicdataset.cv.classification.Mnist;
import ai.djl.basicmodelzoo.cv.classification.ResNetV1;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.zero.Performance;
import ai.djl.zero.RequireZoo;
import java.io.IOException;
import java.util.List;

/** ImageClassification takes an image and classifies the main subject of the image. */
public final class ImageClassification {

    private ImageClassification() {}

    /**
     * Returns a pretrained and ready to use image classification model from our model zoo.
     *
     * @param input the input class between {@link ai.djl.modality.cv.Image}, {@link
     *     java.nio.file.Path}, {@link java.net.URL}, and {@link java.io.InputStream}
     * @param classes what {@link Classes} the image is classified into
     * @param performance the performance tradeoff (see {@link Performance}
     * @param <I> the input type
     * @return the model as a {@link ZooModel} with the {@link Translator} included
     * @throws MalformedModelException if the model zoo model is broken
     * @throws ModelNotFoundException if the model could not be found
     * @throws IOException if the model could not be loaded
     */
    public static <I> ZooModel<I, Classifications> pretrained(
            Class<I> input, Classes classes, Performance performance)
            throws MalformedModelException, ModelNotFoundException, IOException {
        Criteria.Builder<I, Classifications> criteria =
                Criteria.builder()
                        .setTypes(input, Classifications.class)
                        .optApplication(CV.IMAGE_CLASSIFICATION);

        switch (classes) {
            case IMAGENET:
                RequireZoo.mxnet();
                String layers = performance.switchPerformance("18", "50", "152");
                criteria.optGroupId("ai.djl.mxnet")
                        .optArtifactId("resnet")
                        .optFilter("dataset", "imagenet")
                        .optFilter("layers", layers);
                break;
            case DIGITS:
                RequireZoo.basic();
                criteria.optGroupId("ai.djl.zoo")
                        .optArtifactId("mlp")
                        .optFilter("dataset", "mnist");
                break;
            default:
                throw new IllegalArgumentException("Unknown classes");
        }

        return criteria.build().loadModel();
    }

    /**
     * Trains the recommended image classification model on a custom dataset.
     *
     * <p>In order to train on a custom dataset, you must create a custom {@link
     * ImageClassificationDataset} to load your data.
     *
     * @param dataset the data to train with
     * @param performance to determine the desired model tradeoffs
     * @return the model as a {@link ZooModel} with the {@link Translator} included
     * @throws IOException if the dataset could not be loaded
     * @throws TranslateException if the translator has errors
     */
    public static ZooModel<Image, Classifications> train(
            ImageClassificationDataset dataset, Performance performance)
            throws IOException, TranslateException {

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
        List<String> classes = dataset.getClasses();

        Dataset[] splitDataset = dataset.randomSplit(8, 2);
        Dataset trainDataset = splitDataset[0];
        Dataset validateDataset = splitDataset[1];

        // Determine the layers based on performance
        int numLayers = performance.switchPerformance(18, 50, 152);

        Block block =
                ResNetV1.builder()
                        .setImageShape(imageShape)
                        .setNumLayers(numLayers)
                        .setOutSize(classes.size())
                        .build();
        Model model = Model.newInstance("ImageClassification");
        model.setBlock(block);

        TrainingConfig trainingConfig =
                new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss())
                        .addEvaluator(new Accuracy())
                        .addTrainingListeners(TrainingListener.Defaults.basic());

        try (Trainer trainer = model.newTrainer(trainingConfig)) {
            trainer.initialize(new Shape(1).addAll(imageShape));
            EasyTrain.fit(trainer, 35, trainDataset, validateDataset);
        }

        Translator<Image, Classifications> translator = dataset.makeTranslator();
        return new ZooModel<>(model, translator);
    }

    /**
     * The possible classes to classify the images into.
     *
     * <p>The classes available depends on the data that the model was trained with.
     */
    public enum Classes {

        /**
         * Imagenet is a standard dataset of 1000 diverse classes.
         *
         * <p>The dataset can be found at {@link ImageNet}. You can <a
         * href="https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/mxnet/synset.txt">view
         * the list of classes here</a>.
         */
        IMAGENET,

        /**
         * Classify images of the digits 0-9.
         *
         * <p>This contains models trained using the {@link Mnist} dataset.
         */
        DIGITS
    }
}
