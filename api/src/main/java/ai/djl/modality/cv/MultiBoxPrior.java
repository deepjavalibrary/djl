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
import java.util.List;

public class MultiBoxPrior {
    private List<Float> sizes;
    private List<Float> ratios;
    private List<Float> steps;
    private List<Float> offsets;
    private boolean clip;

    public MultiBoxPrior(Builder builder) {
        this.sizes = builder.sizes;
        this.ratios = builder.ratios;
        this.steps = builder.steps;
        this.ratios = builder.ratios;
        this.offsets = builder.offsets;
        this.clip = builder.clip;
    }

    public NDArray generateAnchorBoxes(NDArray input) {
        return input.getNDArrayInternal().multiBoxPrior(sizes, ratios, steps, offsets, clip).head();
    }

    public static final class Builder {
        List<Float> sizes = Arrays.asList(1f);
        List<Float> ratios = Arrays.asList(1f);
        List<Float> steps = Arrays.asList(-1f, -1f);
        List<Float> offsets = Arrays.asList(0.5f, 0.5f);
        boolean clip;

        public Builder setSizes(List<Float> sizes) {
            this.sizes = sizes;
            return this;
        }

        public Builder setRatios(List<Float> ratios) {
            this.ratios = ratios;
            return this;
        }

        public Builder optSteps(List<Float> steps) {
            this.steps = steps;
            return this;
        }

        public Builder optOffsets(List<Float> offsets) {
            this.offsets = offsets;
            return this;
        }

        public void optClip(boolean clip) {
            this.clip = clip;
        }

        public MultiBoxPrior build() {
            return new MultiBoxPrior(this);
        }
    }
}
