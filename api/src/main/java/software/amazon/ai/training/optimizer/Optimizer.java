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

import software.amazon.ai.nn.BlockFactory;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.util.PairList;

/**
 * An optimizer updates a set of parameters based on gradients collected with a {@link
 * software.amazon.ai.training.GradientCollector}.
 */
public interface Optimizer {

    /**
     * Update a {@code PairList} of parameters one step at time. Assumes parameters are on the same
     * context. This will be used when updating parameters locally, not on {@link
     * software.amazon.ai.training.ParameterStore}.
     *
     * @param parameters a {@code PairList} of parameters from network to update
     */
    void updateAllParameters(PairList<String, Parameter> parameters);

    @SuppressWarnings("rawtypes")
    abstract class BaseBuilder<T extends BaseBuilder> {

        protected BlockFactory factory;
        private float rescaleGrad;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

        public T setFactory(BlockFactory factory) {
            this.factory = factory;
            return self();
        }

        public T setRescaleGrad(float rescaleGrad) {
            this.rescaleGrad = rescaleGrad;
            return self();
        }

        public T optWeightDecays(float weightDecays) {
            this.weightDecays = weightDecays;
            return self();
        }

        public T optClipGrad(float clipGrad) {
            this.clipGrad = clipGrad;
            return self();
        }

        public T optBeginNumUpdate(int beginNumUpdate) {
            this.beginNumUpdate = beginNumUpdate;
            return self();
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

        protected abstract T self();
    }
}
