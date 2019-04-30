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
package org.apache.mxnet.engine;

import com.amazon.ai.Block;
import com.amazon.ai.Context;
import com.amazon.ai.inference.Predictor;
import com.amazon.ai.ndarray.NDArray;

public class MxPredictor extends Predictor<NDArray, NDArray> {

    private MxModel model;
    private Context context;
    private Module module;

    MxPredictor(MxModel model, Context context) {
        this.model = model;
        this.context = context;
    }

    @Override
    public NDArray predict(NDArray input) {
        Block network = model.getNetwork();
        network.setInput(input);
        network.forward();
        return network.getOutput();
    }
}
