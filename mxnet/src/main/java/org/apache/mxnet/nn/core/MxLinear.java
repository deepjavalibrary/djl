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
package org.apache.mxnet.nn.core;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.Initializer;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDScopedFactory;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.core.Linear;

public class MxLinear extends MxNNBlock implements Linear {

    private NDArray weight;
    private NDArray bias;

    private int units;
    private int inUnits;

    public MxLinear(int units, int inUnits) {
        this.opName = "FullyConnected";
        this.units = units;
        this.inUnits = inUnits;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getInputShape() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public List<NDArray> getDirectParameters() {
        return Arrays.asList(weight, bias);
    }

    /** {@inheritDoc} */
    @Override
    public void initialize(NDScopedFactory factory, Initializer initializer) {
        weight = factory.create(new DataDesc(new Shape(units, inUnits)));
        bias = factory.create(new DataDesc(new Shape(units)));
        initializer.initialize(getDirectParameters());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray forward(NDArray data) {
        NDList inputs = new NDList(data, weight, bias);
        MxOpParams params = new MxOpParams();
        params.add("num_hidden", "1");
        return forward(inputs, params).get(0);
    }
}
