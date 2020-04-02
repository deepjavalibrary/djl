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
package ai.djl.modality.cv.translator;

import ai.djl.Model;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;

/**
 * A {@link BaseImageTranslator} that post-process the {@link NDArray} into {@link DetectedObjects}
 * with boundaries.
 */
public abstract class ObjectDetectionTranslator extends BaseImageTranslator<DetectedObjects> {

    protected float threshold;
    protected String synsetArtifactName;
    protected List<String> classes;
    protected double imageWidth;
    protected double imageHeight;

    /**
     * Creates the {@link ObjectDetectionTranslator} from the given builder.
     *
     * @param builder the builder for the translator
     */
    public ObjectDetectionTranslator(BaseBuilder<?> builder) {
        super(builder);
        this.threshold = builder.threshold;
        this.synsetArtifactName = builder.synsetArtifactName;
        this.classes = builder.classes;
        this.imageWidth = builder.imageWidth;
        this.imageHeight = builder.imageHeight;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(NDManager manager, Model model) throws IOException {
        if (classes == null) {
            classes = model.getArtifact(synsetArtifactName, Utils::readLines);
        }
    }

    /** The base builder for the object detection translator. */
    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder>
            extends BaseImageTranslator.BaseBuilder<T> {

        protected float threshold = 0.2f;
        protected String synsetArtifactName;
        protected List<String> classes;
        protected double imageWidth;
        protected double imageHeight;

        /** {@inheritDoc} */
        @Override
        protected abstract T self();

        protected void validate() {
            if (synsetArtifactName == null && classes == null) {
                throw new IllegalArgumentException(
                        "You must specify a synset artifact name or classes");
            } else if (synsetArtifactName != null && classes != null) {
                throw new IllegalArgumentException(
                        "You can only specify one of: synset artifact name or classes");
            }
        }

        /**
         * Sets the threshold for prediction accuracy.
         *
         * <p>Predictions below the threshold will be dropped.
         *
         * @param threshold the threshold for the prediction accuracy
         * @return this builder
         */
        public T optThreshold(float threshold) {
            this.threshold = threshold;
            return self();
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
        public T setSynsetArtifactName(String synsetArtifactName) {
            this.synsetArtifactName = synsetArtifactName;
            return self();
        }

        /**
         * Sets the class list.
         *
         * <p>Set either the synset or the classes.
         *
         * @param classes the list of classes
         * @return this builder
         */
        public T setClasses(List<String> classes) {
            this.classes = classes;
            return self();
        }

        /**
         * Sets the optional rescale size.
         *
         * @param imageWidth the width to rescale images to
         * @param imageHeight the height to rescale images to
         * @return this builder
         */
        public T optRescaleSize(double imageWidth, double imageHeight) {
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            return self();
        }

        /**
         * Get threshold.
         *
         * @return threshold
         */
        public float getThreshold() {
            return threshold;
        }

        /**
         * Get symset artifact name.
         *
         * @return name
         */
        public String getSynsetArtifactName() {
            return synsetArtifactName;
        }

        /**
         * Get classes.
         *
         * @return classes
         */
        public List<String> getClasses() {
            return classes;
        }

        /**
         * Get resized image width.
         *
         * @return image width
         */
        public double getImageWidth() {
            return imageWidth;
        }

        /**
         * Get resized image height.
         *
         * @return image height
         */
        public double getImageHeight() {
            return imageHeight;
        }
    }
}
