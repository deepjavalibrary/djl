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

import ai.djl.ndarray.NDList;

/**
 * {@code MultiBoxDetection} is the class that takes the output of a multi-box detection model, and
 * converts it into an NDList that contains the object detections.
 *
 * <p>The output from a Single Shot Detection(SSD) network would be class probabilities, box offset
 * predictions, and the generated anchor boxes. Once out-of-boundary boxes are eliminated, and boxes
 * with scores lower than the threshold are removed, we will ideally have a small number of
 * candidates for each object in the image. Since anchor boxes are generated in multiple scales
 * around each pixel by {@link MultiBoxPrior}, there are bound to be multiple boxes around each
 * object which have a score greater than the threshold. We use Non-Maximum Suppression(NMS) to
 * choose one box that is most likely to fit the object in the image.
 *
 * <p>{@code MultiBoxDetection} handles all of these tasks, and returns an {@link NDList} with a
 * single {@link ai.djl.ndarray.NDArray} of {@link ai.djl.ndarray.types.Shape} (batch_size, Number
 * of generated anchor boxes, 6). For each generated anchor box, there is an {@link
 * ai.djl.ndarray.NDArray} of {@link ai.djl.ndarray.types.Shape} (6,). The values in each of those
 * arrays represent the following: {@code [class, score, x_min, y_min, x_max, y_max]}. The {@code
 * class} is set to -1 for boxes that are removed or classified as background. The {@code score} is
 * the confidence with which the model thinks the box contains an object of the specified {@code
 * class}, and the other four values represent the normalised co-ordinates of the box.
 */
public class MultiBoxDetection {

    private boolean clip;
    private float threshold;
    private int backgroundId;
    private float nmsThreshold;
    private boolean forceSuppress;
    private int nmsTopK;

    /**
     * Creates a new instance of {@code MultiBoxDetection} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     */
    public MultiBoxDetection(Builder builder) {
        this.clip = builder.clip;
        this.threshold = builder.threshold;
        this.backgroundId = builder.backgroundId;
        this.nmsThreshold = builder.nmsThreshold;
        this.forceSuppress = builder.forceSuppress;
        this.nmsTopK = builder.nmsTopK;
    }

    /**
     * Converts multi-box detection predictions.
     *
     * @param inputs a NDList of (class probabilities, box predictions, and anchors) in that order
     * @return an {@link NDList} with a single {@link ai.djl.ndarray.NDArray} of {@link
     *     ai.djl.ndarray.types.Shape} (batch_size, Number of generated anchor boxes, 6). For each
     *     generated anchor box, there is an {@link ai.djl.ndarray.NDArray} of {@link
     *     ai.djl.ndarray.types.Shape} (6,). The values in each of those arrays represent the
     *     following: {@code [class, score, x_min, y_min, x_max, y_max]}
     */
    public NDList detection(NDList inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException(
                    "NDList must contain class probabilities, box predictions, and anchors");
        }
        return inputs.head()
                .getNDArrayInternal()
                .multiBoxDetection(
                        inputs,
                        clip,
                        threshold,
                        backgroundId,
                        nmsThreshold,
                        forceSuppress,
                        nmsTopK);
    }

    /**
     * Creates a builder to build a {@code MultiBoxDetection}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link MultiBoxDetection} object. */
    public static final class Builder {

        boolean clip = true;
        private float threshold = 0.01f;
        int backgroundId;
        private float nmsThreshold = 0.5f;
        boolean forceSuppress;
        private int nmsTopK = -1;

        Builder() {}

        /**
         * Sets the boolean parameter that indicates whether to clip out-of-boundary boxes. It is
         * set to {@code true} by default.
         *
         * @param clip whether to clip out-of-boundary boxes
         * @return this {@code Builder}
         */
        public Builder optClip(boolean clip) {
            this.clip = clip;
            return this;
        }

        /**
         * Sets the boolean parameter that indicates whether to suppress all detections regardless
         * of class_id. It is set to {@code false} by default.
         *
         * @param forceSuppress whether to suppress all detections regardless of class_id
         * @return this {@code Builder}
         */
        public Builder optForceSuppress(boolean forceSuppress) {
            this.forceSuppress = forceSuppress;
            return this;
        }

        /**
         * Sets the class ID for the background. Defaults to 0.
         *
         * @param backgroundId the class ID for the background
         * @return this {@code Builder}
         */
        public Builder optBackgroundId(int backgroundId) {
            this.backgroundId = backgroundId;
            return this;
        }

        /**
         * Sets the boolean parameter that indicates whether to clip out-of-boundary boxes. Defaults
         * to -1 which implies that there is no limit.
         *
         * @param nmsTopK whether to clip out-of-boundary boxes
         * @return this {@code Builder}
         */
        public Builder optNmsTopK(int nmsTopK) {
            this.nmsTopK = nmsTopK;
            return this;
        }

        /**
         * Sets the threshold score for a detection to be a positive prediction. Defaults to 0.01.
         *
         * @param threshold the threshold score for a detection to be a positive prediction
         * @return this {@code Builder}
         */
        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * Sets the non-maximum suppression(NMS) threshold. Defaults to 0.5.
         *
         * @param nmsThreshold the non-maximum suppression(NMS) threshold
         * @return this {@code Builder}
         */
        public Builder optNmsThreshold(float nmsThreshold) {
            this.nmsThreshold = nmsThreshold;
            return this;
        }

        /**
         * Builds a {@link MultiBoxDetection} block.
         *
         * @return the {@link MultiBoxDetection} block
         */
        public MultiBoxDetection build() {
            return new MultiBoxDetection(this);
        }
    }
}
