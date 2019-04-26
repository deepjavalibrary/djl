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
package com.amazon.ai.training;

import com.amazon.ai.Block;
import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.NDArray;

public class Trainer {

    private Model model;
    private Estimator estimator;
    private Optimizer optimizer;
    private ModelSaver modelSaver;

    public static Trainer newInstance(Model model) {
        return newInstance(model, Context.defaultContext());
    }

    public static Trainer newInstance(Model model, Context context) {
        return Engine.getInstance().newTrainer(model, context);
    }

    public Trainer(Model model) {
        this.model = model;
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

    public ModelSaver getModelSaver() {
        return modelSaver;
    }

    public void setModelSaver(ModelSaver modelSaver) {
        this.modelSaver = modelSaver;
    }

    public NDArray train(NDArray array) {
        Block graph = model.getNetwork();
        graph.setInput(array);
        graph.forward();

        array = graph.getOutput();

        return array;
    }

    public void checkpoint() {
        modelSaver.save(model);
    }
}
