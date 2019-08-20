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

package software.amazon.ai.training.optimizer;

import software.amazon.ai.Parameter;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.training.optimizer.learningrate.LrTracker;
import software.amazon.ai.util.PairList;

/** An NAG optimizer. Build with {@link Nag.Builder}. */
public interface Nag extends Optimizer {

    class Builder {

        PairList<String, Parameter> parameters;
        float rescaleGrad;
        float weightDecays;
        float clipGrad = -1;
        LrTracker lrTracker;
        int beginNumUpdate;
        float momentum;

        public Builder(PairList<String, Parameter> parameters) {
            this.parameters = parameters;
        }

        public Nag.Builder setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return this;
        }

        public Nag.Builder setWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return this;
        }

        public Nag.Builder setClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return this;
        }

        public Nag.Builder setLrTracker(LrTracker lrTracker) {
            this.lrTracker = lrTracker;
            return this;
        }

        public Nag.Builder setBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return this;
        }

        public Nag.Builder setMomentum(float momentum) {
            this.momentum = momentum;
            return this;
        }

        public Nag build() {
            return Engine.getInstance()
                    .getNNIndex()
                    .nag(
                            parameters,
                            rescaleGrad,
                            weightDecays,
                            clipGrad,
                            lrTracker,
                            beginNumUpdate,
                            momentum);
        }
    }
}
