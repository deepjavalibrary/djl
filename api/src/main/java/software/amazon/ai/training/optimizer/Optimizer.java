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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.ParameterServer;
import software.amazon.ai.util.PairList;

/** MXNet helper containing base implementations for optimizers. */
public abstract class Optimizer {

    protected float rescaleGrad;
    protected float clipGrad;
    private float weightDecays;
    private int numUpdate;
    private boolean statesInitialized;
    private Map<Integer, Integer> updateCounts = new ConcurrentHashMap<>();

    public Optimizer(BaseBuilder<?> builder) {
        this.rescaleGrad = builder.getRescaleGrad();
        this.weightDecays = builder.getWeightDecays();
        this.clipGrad = builder.getClipGrad();
        this.numUpdate = builder.getBeginNumUpdate();

        if (rescaleGrad == 0) {
            throw new IllegalArgumentException("The rescaleGrad should be set");
        }
    }

    /**
     * Update a {@code PairList} of parameters one step at time. Assumes parameters are on the same
     * device. This will be used when updating parameters locally, not on {@link ParameterServer}.
     *
     * @param parameters a {@code PairList} of parameters from network to update
     */
    public void updateAllParameters(PairList<String, Parameter> parameters) {
        if (!statesInitialized) {
            // ensure when create state is over ridden, statesCreated is updated
            statesInitialized = initializeStates(parameters);
        }
        for (int i = 0; i < parameters.size(); i++) {
            Parameter param = parameters.get(i).getValue();
            if (param.requireGradient()) {
                NDArray paramArray = param.getArray();
                NDArray grad = paramArray.getGradient();
                update(i, paramArray, grad);
            }
        }
    }

    protected float getWeightDecay(int index) {
        return weightDecays;
    }

    protected int updateCount(int index) {
        int count = updateCounts.compute(index, (key, val) -> (val == null) ? numUpdate : val + 1);
        numUpdate = Math.max(numUpdate, count);
        return count;
    }

    // TODO: make this protected after integrate with PS store
    public abstract void update(int index, NDArray weight, NDArray grad);

    protected abstract boolean initializeStates(PairList<String, Parameter> parameters);

    @SuppressWarnings("rawtypes")
    public abstract static class BaseBuilder<T extends BaseBuilder> {

        private float rescaleGrad;
        private float weightDecays;
        private float clipGrad = -1;
        private int beginNumUpdate;

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
