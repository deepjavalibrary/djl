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
package software.amazon.ai.modality.cv;

import software.amazon.ai.ndarray.NDList;

public class MultiBoxTarget {
    int minNegativeSamples;
    private float iouThreshold;
    private float negativeMiningRatio;
    private float ignoreLabel;
    private float negativeMiningThreshold;

    public MultiBoxTarget(Builder builder) {
        this.minNegativeSamples = builder.minNegativeSamples;
        this.iouThreshold = builder.iouThreshold;
        this.negativeMiningThreshold = builder.negativeMiningThreshold;
        this.negativeMiningRatio = builder.negativeMinigRatio;
        this.ignoreLabel = builder.ignoreLabel;
    }

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

    public static final class Builder {
        int minNegativeSamples;
        float iouThreshold = 0.5f;
        float ignoreLabel = -1;
        float negativeMiningThreshold = 0.5f;
        float negativeMinigRatio = -1;

        public Builder optMinNegativeSamples(int minNegativeSamples) {
            this.minNegativeSamples = minNegativeSamples;
            return this;
        }

        public Builder optIouThreshold(float iouThreshold) {
            this.iouThreshold = iouThreshold;
            return this;
        }

        public Builder optIgnoreLabel(float ignoreLabel) {
            this.ignoreLabel = ignoreLabel;
            return this;
        }

        public Builder optNegativeMiningThreshold(float negativeMiningThreshold) {
            this.negativeMiningThreshold = negativeMiningThreshold;
            return this;
        }

        public Builder optNegativeMinigRatio(float negativeMinigRatio) {
            this.negativeMinigRatio = negativeMinigRatio;
            return this;
        }

        public MultiBoxTarget build() {
            return new MultiBoxTarget(this);
        }
    }
}
