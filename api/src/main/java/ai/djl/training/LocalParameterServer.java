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
package ai.djl.training;

import ai.djl.ndarray.NDArray;
import ai.djl.training.optimizer.Optimizer;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LocalParameterServer implements ParameterServer {

    private Optimizer optimizer;
    private Map<String, NDArray[]> gradMap;

    public LocalParameterServer(Optimizer optimizer) {
        this.optimizer = optimizer;
        gradMap = new ConcurrentHashMap<>();
    }

    @Override
    public void init(String parameterId, NDArray[] value) {}

    @Override
    public void push(String parameterId, NDArray[] grads, int priority) {
        NDArray[] oldGrads = gradMap.put(parameterId, grads);
        if (oldGrads != null) {
            Arrays.stream(oldGrads).forEach(NDArray::close);
        }
    }

    @Override
    public void pull(String parameterId, NDArray[] weights, int priority) {
        NDArray[] grads = gradMap.get(parameterId);
        for (int i = 0; i < grads.length; ++i) {
            optimizer.update(parameterId, weights[i], grads[i]);
        }
    }

    @Override
    public void close() {}
}
