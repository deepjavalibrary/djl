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

public class MultiBoxDetection {
    private boolean clip;
    private float threshold;
    private int backgroundId;
    private float nmsThreashold;
    private boolean forceSuppress;
    private int nmsTopK;

    public MultiBoxDetection(Builder builder) {
        this.clip = builder.clip;
        this.threshold = builder.threshold;
        this.backgroundId = builder.backgroundId;
        this.nmsThreashold = builder.nmsThreashold;
        this.forceSuppress = builder.forceSuppress;
        this.nmsTopK = builder.nmsTopK;
    }

    public NDList detection(NDList inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException(
                    "NDList must contain class probabilites, box predictions, and anchors");
        }
        return inputs.head()
                .getNDArrayInternal()
                .multiBoxDetection(
                        inputs,
                        clip,
                        threshold,
                        backgroundId,
                        nmsThreashold,
                        forceSuppress,
                        nmsTopK);
    }

    public static final class Builder {
        boolean clip = true;
        private float threshold = 0.00999999978f;
        int backgroundId;
        private float nmsThreashold = 0.5f;
        boolean forceSuppress;
        private int nmsTopK = -1;

        public Builder optClip(boolean clip) {
            this.clip = clip;
            return this;
        }

        public Builder optForceSuppress(boolean forceSuppress) {
            this.forceSuppress = forceSuppress;
            return this;
        }

        public Builder optBackgroundId(int backgroundId) {
            this.backgroundId = backgroundId;
            return this;
        }

        public Builder optNmsTopK(int nmsTopK) {
            this.nmsTopK = nmsTopK;
            return this;
        }

        public Builder optThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder optNmsThreashold(float nmsThreashold) {
            this.nmsThreashold = nmsThreashold;
            return this;
        }

        public MultiBoxDetection build() {
            return new MultiBoxDetection(this);
        }
    }
}
