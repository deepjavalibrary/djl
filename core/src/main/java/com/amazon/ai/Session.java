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

package com.amazon.ai;

import com.amazon.ai.util.PairList;

public class Session {

    private Model model;
    private String[] inputParamNames;
    private boolean forTraining;

    private Transformer transformer;
    private Estimator estimator;
    private Optimizer optimizer;

    Session(Model model, String[] inputParamNames, boolean forTraining) {
        this.model = model;
        this.inputParamNames = inputParamNames;
        this.forTraining = forTraining;
    }

    public static Session forInference(Model model, String[] inputParamNames) {
        return new Session(model, inputParamNames, false);
    }

    public static Session forTraining(Model model, String[] inputParamNames) {
        return new Session(model, inputParamNames, true);
    }

    public String[] getInputParamNames() {
        return inputParamNames;
    }

    public boolean isForTraining() {
        return forTraining;
    }

    public Transformer getTransformer() {
        return transformer;
    }

    public void setTransformer(Transformer transformer) {
        this.transformer = transformer;
    }

    public Estimator getEstimator() {
        return estimator;
    }

    public void setEstimator(Estimator estimator) {
        this.estimator = estimator;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public void start() {}

    public Tensor process(Tensor tensor) {
        if (transformer != null) {
            tensor = transformer.preprocess(tensor);
        }

        Graph graph = model.getGraph();
        graph.call(tensor);

        tensor = graph.getOutput();

        if (transformer != null) {
            tensor = transformer.postprocess(tensor);
        }
        return tensor;
    }

    public PairList<String, Parameter> getParameters() {
        return null;
    }

    public void close() {}
}
