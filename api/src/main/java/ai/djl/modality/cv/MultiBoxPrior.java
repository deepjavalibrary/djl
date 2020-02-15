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

import ai.djl.ndarray.NDArray;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * {@code MultiBoxPrior} is the class that generates anchor boxes that act as priors for object
 * detection.
 *
 * <p>Object detection algorithms usually sample a large number of regions in the input image,
 * determine whether these regions contain objects of interest, and adjust the edges of the regions
 * so as to predict the ground-truth bounding box of the target more accurately. Different models
 * may use different region sampling methods. These bounding boxes are also called anchor boxes.
 *
 * <p>{@code MultiBoxPrior} generates these anchor boxes, based on the required sizes and aspect
 * ratios and returns an {@link NDArray} of {@link ai.djl.ndarray.types.Shape} (1, Number of anchor
 * boxes, 4). Anchor boxes need not be generated separately for the each example in the batch. One
 * set of anchor boxes per batch is sufficient.
 *
 * <p>The number of anchor boxes generated depends on the number of sizes and aspect ratios. If the
 * number of sizes is \(n\) and the number of ratios is \(m\), the total number of boxes generated
 * per pixel is \(n + m - 1\).
 */
public class MultiBoxPrior {

    private List<Float> sizes;
    private List<Float> ratios;
    private List<Float> steps;
    private List<Float> offsets;
    private boolean clip;

    /**
     * Creates a new instance of {@code MultiBoxPrior} with the arguments from the given {@link
     * Builder}.
     *
     * @param builder the {@link Builder} with the necessary arguments
     */
    public MultiBoxPrior(Builder builder) {
        this.sizes = builder.sizes;
        this.ratios = builder.ratios;
        this.steps = builder.steps;
        this.offsets = builder.offsets;
        this.clip = builder.clip;
    }

    /**
     * Generates the anchorBoxes array in the input's manager and device.
     *
     * @param input the input whose manager and device to put the generated boxes in
     * @return the generated boxes
     */
    public NDArray generateAnchorBoxes(NDArray input) {
        return input.getNDArrayInternal().multiBoxPrior(sizes, ratios, steps, offsets, clip).head();
    }

    /**
     * Creates a builder to build a {@code MultiBoxPrior}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link MultiBoxPrior} object. */
    public static final class Builder {

        List<Float> sizes = Collections.singletonList(1f);
        List<Float> ratios = Collections.singletonList(1f);
        List<Float> steps = Arrays.asList(-1f, -1f);
        List<Float> offsets = Arrays.asList(0.5f, 0.5f);
        boolean clip;

        Builder() {}

        /**
         * Sets the sizes of the anchor boxes to be generated around each pixel.
         *
         * @param sizes the size of the anchor boxes generated around each pixel
         * @return this {@code Builder}
         */
        public Builder setSizes(List<Float> sizes) {
            this.sizes = sizes;
            return this;
        }

        /**
         * Sets the aspect ratios of the anchor boxes to be generated around each pixel.
         *
         * @param ratios the aspect ratios of the anchor boxes to be generated around each pixel
         * @return this {@code Builder}
         */
        public Builder setRatios(List<Float> ratios) {
            this.ratios = ratios;
            return this;
        }

        /**
         * Sets the step across \(x\) and \(y\) dimensions. Defaults to -1 across both dimensions.
         *
         * @param steps the step across \(x\) and \(y\) dimensions
         * @return this {@code Builder}
         */
        public Builder optSteps(List<Float> steps) {
            this.steps = steps;
            return this;
        }

        /**
         * Sets the value of the center-box offsets across \(x\) and \(y\) dimensions. Defaults to
         * 0.5 across both dimensions.
         *
         * @param offsets the value of the center-box offsets across \(x\) and \(y\) dimensions
         * @return this {@code Builder}
         */
        public Builder optOffsets(List<Float> offsets) {
            this.offsets = offsets;
            return this;
        }

        /**
         * Sets the boolean parameter that indicates whether to clip out-of-boundary boxes. It is
         * set to {@code false} by default.
         *
         * @param clip whether to clip out-of-boundary boxes
         * @return this {@code Builder}
         */
        public Builder optClip(boolean clip) {
            this.clip = clip;
            return this;
        }

        /**
         * Builds a {@link MultiBoxPrior} block.
         *
         * @return the {@link MultiBoxPrior} block
         */
        public MultiBoxPrior build() {
            return new MultiBoxPrior(this);
        }
    }
}
