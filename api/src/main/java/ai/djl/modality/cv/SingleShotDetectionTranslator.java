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
package ai.djl.modality.cv;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * A {@link ImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects} with
 * boundaries.
 */
public class SingleShotDetectionTranslator extends ImageTranslator<DetectedObjects> {

    private float threshold;
    private String synsetArtifactName;
    private List<String> classes;
    private double imageWidth;
    private double imageHeight;

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SingleShotDetectionTranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.synsetArtifactName = builder.synsetArtifactName;
        this.classes = builder.classes;
        this.imageWidth = builder.imageWidth;
        this.imageHeight = builder.imageHeight;
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();
        if (classes == null) {
            classes = model.getArtifact(synsetArtifactName, Utils::readLines);
        }

        float[] classIds = list.get(0).toFloatArray();
        float[] probabilities = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < classIds.length; ++i) {
            int classId = (int) classIds[i];
            double probability = probabilities[i];
            // classId starts from 0, -1 means background
            if (classId >= 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                // rescale box coordinates by imageWidth and imageHeight
                double x = imageWidth > 0 ? box[0] / imageWidth : box[0];
                double y = imageHeight > 0 ? box[1] / imageHeight : box[1];
                double w = imageWidth > 0 ? box[2] / imageWidth - x : box[2] - x;
                double h = imageHeight > 0 ? box[3] / imageHeight - y : box[3] - y;

                Rectangle rect = new Rectangle(x, y, w, h);
                retNames.add(className);
                retProbs.add(probability);
                retBB.add(rect);
            }
        }

        return new DetectedObjects(retNames, retProbs, retBB);
    }

    /** The builder for SSD translator. */
    public static class Builder extends BaseBuilder<Builder> {

        private float threshold = 0.2f;
        private String synsetArtifactName;
        private List<String> classes;
        private double imageWidth;
        private double imageHeight;

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for the prediction accuracy
         * @return this builder
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Sets the name for the synset.
         *
         * <p>Synset is used to convert the prediction classes to their actual names.
         *
         * <p>Set either the synset or the classes.
         *
         * @param synsetArtifactName the name of synset
         * @return this builder
         */
        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        /**
         * Sets the class list.
         *
         * <p>Set either the synset or the classes.
         *
         * @param classes the list of classes
         * @return this builder
         */
        public Builder setClasses(List<String> classes) {
            this.classes = classes;
            return this;
        }

        /**
         * Sets the optional rescale size.
         *
         * @param imageWidth the width to rescale images to
         * @param imageHeight the height to rescale images to
         * @return this builder
         */
        public Builder optRescaleSize(double imageWidth, double imageHeight) {
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the translator.
         *
         * @return the new translator
         */
        public SingleShotDetectionTranslator build() {
            if (synsetArtifactName == null && classes == null) {
                throw new IllegalArgumentException(
                        "You must specify a synset artifact name or classes");
            } else if (synsetArtifactName != null && classes != null) {
                throw new IllegalArgumentException(
                        "You can only specify one of: synset artifact name or classes");
            }
            return new SingleShotDetectionTranslator(this);
        }
    }
}
