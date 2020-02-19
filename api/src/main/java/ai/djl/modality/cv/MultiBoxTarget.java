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
 * {@code MultiBoxTarget} is the class that computes the training targets for training a Single Shot
 * Detection (SSD) models.
 *
 * <p>The output from a Single Shot Detection (SSD) network would be class probabilities, box offset
 * predictions, and the generated anchor boxes. The labels contain a class label and the bounding
 * box for each object in the image. The generated anchor boxes are each a prior, and need loss
 * computed for each of them. This requires that we assign a ground truth box to every one of them.
 *
 * <p>{@code MultiBoxTarget} takes an {@link NDList} containing (anchor boxes, labels, class
 * predictions) in that order. It computes the Intersection-over-Union (IoU) of each anchor box
 * against every ground-truth box. For every anchor box, it assigns a ground-truth box with maximum
 * IoU with respect to the anchor box if the IoU is greater than a given threshold. Once a
 * ground-truth box is assigned for each anchor box, it computes the offset of each anchor box with
 * respect to it's assigned ground-truth box.
 *
 * <p>{@code MultiBoxTarget} handles these tasks and returns an {@link NDList} containing (Bounding
 * box offsets, bounding box masks, class labels). Bounding box offsets and class labels are
 * computed as above. Bounding box masks is a mask array that contains either a 0 or 1, with the 0s
 * corresponding to the anchor boxes whose IoUs with the ground-truth boxes were less than the given
 * threshold.
 */
public class MultiBoxTarget {

    int minNegativeSamples;
    private float iouThreshold;
    private float negativeMiningRatio;
    private float ignoreLabel;
    private float negativeMiningThreshold;

    /**
     * Creates a new instance of {@code MultiBoxTarget} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     */
    public MultiBoxTarget(Builder builder) {
        this.minNegativeSamples = builder.minNegativeSamples;
        this.iouThreshold = builder.iouThreshold;
        this.negativeMiningThreshold = builder.negativeMiningThreshold;
        this.negativeMiningRatio = builder.negativeMinigRatio;
        this.ignoreLabel = builder.ignoreLabel;
    }

    /**
     * Computes multi-box training targets.
     *
     * @param inputs a NDList of (anchors, labels, and class prediction) in that order
     * @return an {@link NDList} containing (Bounding box offsets, bounding box masks, class labels)
     */
    public NDList target(NDList inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException(
                    "NDList must contain anchors, labels, and class predictions");
        }
        return inputs.head()
                .getNDArrayInternal()
                .multiBoxTarget(
                        inputs,
                        iouThreshold,
                        ignoreLabel,
                        negativeMiningRatio,
                        negativeMiningThreshold,
                        minNegativeSamples);
    }

    /**
     * Creates a builder to build a {@code MultiBoxTarget}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link MultiBoxTarget} object. */
    public static final class Builder {

        int minNegativeSamples;
        float iouThreshold = 0.5f;
        float ignoreLabel = -1;
        float negativeMiningThreshold = 0.5f;
        float negativeMinigRatio = -1;

        Builder() {}

        /**
         * Sets the minimum number of negative samples.
         *
         * @param minNegativeSamples the minimum number of negative samples
         * @return this {@code Builder}
         */
        public Builder optMinNegativeSamples(int minNegativeSamples) {
            this.minNegativeSamples = minNegativeSamples;
            return this;
        }

        /**
         * Sets the anchor-GroundTruth overlap threshold to be regarded as a positive match.
         *
         * @param iouThreshold the anchor-GroundTruth overlap threshold to be regarded as a positive
         *     match
         * @return this {@code Builder}
         */
        public Builder optIouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        /**
         * Sets the label for ignored anchors. Defaults to -1.
         *
         * @param ignoreLabel the label for ignored anchors
         * @return this {@code Builder}
         */
        public Builder optIgnoreLabel(float ignoreLabel) {
            this.ignoreLabel = ignoreLabel;
            return this;
        }

        /**
         * Sets the threshold used for negative mining.
         *
         * @param negativeMiningThreshold the threshold used for negative mining
         * @return this {@code Builder}
         */
        public Builder optNegativeMiningThreshold(float negativeMiningThreshold) {
            this.negativeMiningThreshold = negativeMiningThreshold;
            return this;
        }

        /**
         * Sets the max negative to positive samples ratio. Use -1 to disable mining. Defaults to
         * -1.
         *
         * @param negativeMinigRatio the max negative to positive samples ratio
         * @return this {@code Builder}
         */
        public Builder optNegativeMinigRatio(float negativeMinigRatio) {
            this.negativeMinigRatio = negativeMinigRatio;
            return this;
        }

        /**
         * Builds a {@link MultiBoxTarget} block.
         *
         * @return the {@link MultiBoxTarget} block
         */
        public MultiBoxTarget build() {
            return new MultiBoxTarget(this);
        }
    }
}
