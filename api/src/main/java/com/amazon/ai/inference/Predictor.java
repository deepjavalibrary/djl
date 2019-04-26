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
package com.amazon.ai.inference;

import com.amazon.ai.Block;
import com.amazon.ai.Context;
import com.amazon.ai.Model;
import com.amazon.ai.engine.Engine;
import com.amazon.ai.ndarray.NDArray;

public class Predictor {

    protected Model model;
    protected Context context;

    public static Predictor newInstance(Model model) {
        return newInstance(model, Context.defaultContext());
    }

    public static Predictor newInstance(Model model, Context context) {
        return Engine.getInstance().newPredictor(model, context);
    }

    public Predictor(Model model, Context context) {
        this.model = model;
        this.context = context;
    }

    public Model getModel() {
        return model;
    }

    public Context getContext() {
        return context;
    }

    public NDArray predict(NDArray array) {
        Block network = model.getNetwork();
        network.setInput(array);
        network.forward();

        return network.getOutput();
    }
}
