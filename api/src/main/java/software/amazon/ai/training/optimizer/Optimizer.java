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
import software.amazon.ai.util.PairList;

/**
 * An optimizer updates a set of parameters based on gradients collected with a {@link
 * software.amazon.ai.training.Gradient.Collector}.
 */
public interface Optimizer {

    PairList<String, Parameter> getParameters();

    void step();

    @SuppressWarnings("rawtypes")
    abstract class BaseBuilder<B extends BaseBuilder> {

        private PairList<String, Parameter> parameters;
        private float rescaleGrad;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

        BaseBuilder(PairList<String, Parameter> parameters) {
            this.parameters = parameters;
        }

        public B setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return self();
        }

        public B optWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return self();
        }

        public B optClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return self();
        }

        public B optBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return self();
        }

        public PairList<String, Parameter> getParameters() {
            return parameters;
        }

        public float getRescaleGrad() {
            return rescaleGrad;
        }

        public float getWeightDecays() {
            return weightDecays;
        }

        public float getClipGrad() {
            return clipGrad;
        }

        public int getBeginNumUpdate() {
            return beginNumUpdate;
        }

        abstract B self();
    }
}
