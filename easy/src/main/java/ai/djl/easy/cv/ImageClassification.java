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
package ai.djl.easy.cv;

import ai.djl.Application.CV;
import ai.djl.MalformedModelException;
import ai.djl.easy.Performance;
import ai.djl.easy.RequireZoo;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import java.io.IOException;

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
     * @return a pretrained and ready to use model from our model zoo
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
                criteria.optGroupId("ai.djl.mxnet")
                        .optArtifactId("resnet")
                        .optFilter("dataset", "imagenet");
                switch (performance) {
                    case FAST:
                        criteria.optFilter("layers", "18");
                        break;
                    case BALANCED:
                        criteria.optFilter("layers", "50");
                        break;
                    case ACCURATE:
                        criteria.optFilter("layers", "152");
                        break;
                    default:
                        throw new IllegalArgumentException("Unknown performance");
                }
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

        return ModelZoo.loadModel(criteria.build());
    }

    /*
    I am leaving this commented out as an example of what the DJL-Easy train API should look like.
    public static <I> ZooModel<I, Classifications> train(Class<I> input, Dataset dataset, Performance performance) {
        throw new UnsupportedOperationException("Not yet implemented");
    }
     */

    /**
     * The possible classes to classify the images into.
     *
     * <p>The classes available depends on the data that the model was trained with.
     */
    public enum Classes {

        /**
         * Imagenet is a standard dataset of 1000 diverse classes.
         *
         * <p>The dataset can be found at {@link ai.djl.basicdataset.ImageNet}. You can <a
         * href="https://djl-ai.s3.amazonaws.com/mlrepo/model/cv/image_classification/ai/djl/mxnet/synset.txt">view
         * the list of classes here</a>.
         */
        IMAGENET,

        /**
         * Classify images of the digits 0-9.
         *
         * <p>This contains models trained using the {@link ai.djl.basicdataset.Mnist} dataset.
         */
        DIGITS
    }
}
