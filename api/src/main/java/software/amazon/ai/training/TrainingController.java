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

package software.amazon.ai.training;

import java.util.List;
import java.util.stream.Collectors;
import software.amazon.ai.engine.Engine;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDArrays;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.training.optimizer.Optimizer;
import software.amazon.ai.util.Pair;
import software.amazon.ai.util.PairList;

public class TrainingController implements AutoCloseable {
    private PairList<String, Parameter> parameters;
    private Optimizer optimizer;
    private ParameterStore parameterStore;
    private boolean gradientsChecked;
    private boolean updateOnParameterStore;

    public TrainingController(PairList<String, Parameter> parameters, Optimizer optimizer) {
        this(parameters, optimizer, false, false);
    }

    public TrainingController(
            PairList<String, Parameter> parameters,
            Optimizer optimizer,
            boolean updateOnParameterStore,
            boolean aggregateOnGPU) {
        this.parameters = parameters;
        this.optimizer = optimizer;
        this.updateOnParameterStore = updateOnParameterStore;
        parameterStore = Engine.getInstance().newParameterStore(optimizer, aggregateOnGPU);
    }

    /** Makes one step of parameter update. */
    public void step() {
        if (!gradientsChecked) {
            checkGradients();
        }
        if (updateOnParameterStore) {
            reduceGradientsOnParameterStore();
            updateOnParameterStore();
        } else {
            optimizer.updateAllParameters(parameters);
        }
    }

    /**
     * Check if all gradients are zeros, prevent users from calling step() without running {@code
     * backward}.
     */
    private void checkGradients() {
        List<NDArray> grads =
                parameters
                        .stream()
                        .map(pair -> pair.getValue().getArray().getGradient())
                        .collect(Collectors.toList());
        NDArray gradSum = NDArrays.stack(grads.stream().map(NDArray::sum).toArray(NDArray[]::new));
        float[] sums = gradSum.sum().toFloatArray();
        float sum = 0f;
        for (float num : sums) {
            sum += num;
        }
        if (sum == 0f) {
            throw new IllegalStateException(
                    "Gradient values are all zeros, please call gradientCollector.backward() on"
                            + "your target NDArray (usually loss), before calling step() ");
        }
        gradientsChecked = true;
    }

    void reduceGradientsOnParameterStore() {
        for (int i = 0; i < parameters.size(); i++) {
            // TODO: handle gradient from multiple contexts
            NDArray gradient = parameters.get(i).getValue().getArray().getGradient();
            parameterStore.push(i, gradient);
        }
    }

    void updateOnParameterStore() {
        for (int i = 0; i < parameters.size(); i++) {
            // TODO: handle update from multiple contexts
            NDArray paramArray = parameters.get(i).getValue().getArray();
            parameterStore.pull(i, paramArray);
        }
    }

    public PairList<String, Parameter> getParameters() {
        return parameters;
    }

    @Override
    public void close() {
        parameterStore.close();
        for (Pair<String, Parameter> pair : parameters) {
            pair.getValue().close();
        }
    }
}
