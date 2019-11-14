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
package ai.djl.mxnet.zoo.cv.objectdetection;

import ai.djl.Model;
import ai.djl.modality.cv.BoundingBox;
import ai.djl.modality.cv.DetectedObjects;
import ai.djl.modality.cv.ImageTranslator;
import ai.djl.modality.cv.Rectangle;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/** The translator for {@link SingleShotDetectionModelLoader}. */
public class SingleShotDetectionTranslator extends ImageTranslator<DetectedObjects> {

    private float threshold;
    private String synsetArtifactName;

    /**
     * Creates the SSD translator from the given builder.
     *
     * @param builder the builder for the translator
     */
    public SingleShotDetectionTranslator(Builder builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.synsetArtifactName = builder.synsetArtifactName;
    }

    /** {@inheritDoc} */
    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) throws IOException {
        Model model = ctx.getModel();
        List<String> classes = model.getArtifact(synsetArtifactName, Utils::readLines);

        float[] classIds = list.get(0).toFloatArray();
        float[] probabilities = list.get(1).toFloatArray();
        NDArray boundingBoxes = list.get(2);

        List<String> retNames = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        for (int i = 0; i < classIds.length; ++i) {
            int classId = (int) classIds[i];
            double probability = probabilities[i];
            if (classId > 0 && probability > threshold) {
                if (classId >= classes.size()) {
                    throw new AssertionError("Unexpected index: " + classId);
                }
                String className = classes.get(classId);
                float[] box = boundingBoxes.get(i).toFloatArray();
                double x = box[0] / 512;
                double y = box[1] / 512;
                double w = box[2] / 512 - x;
                double h = box[3] / 512 - y;

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

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for the prediction accuracy
         * @return the builder
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
         * @param synsetArtifactName the name of synset
         * @return the builder
         */
        public Builder setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        public SingleShotDetectionTranslator build() {
            if (synsetArtifactName == null) {
                throw new IllegalArgumentException("You must specify a synset artifact name");
            }
            return new SingleShotDetectionTranslator(this);
        }
    }
}
