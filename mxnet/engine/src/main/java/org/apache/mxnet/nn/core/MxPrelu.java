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

import java.util.Collections;
import java.util.List;
import org.apache.mxnet.engine.MxOpParams;
import org.apache.mxnet.nn.MxNNBlock;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.nn.Parameter;
import software.amazon.ai.nn.ParameterType;
import software.amazon.ai.nn.core.Prelu;
import software.amazon.ai.util.PairList;

public class MxPrelu extends MxNNBlock implements Prelu {

    private Parameter alpha;

    public MxPrelu() {
        this.opName = "LeakyReLU";
        this.alpha = new Parameter("alpha", this, ParameterType.OTHER);
    }

    @Override
    public Shape getOutputShape(Shape... inputs) {
        return inputs[0];
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(alpha);
    }

    @Override
    public void beforeInitialize(NDList inputs) {}

    @Override
    public Shape getParameterShape(String name, NDList inputs) {
        switch (name) {
            case "alpha":
                // TODO: This should return Shape()
                return new Shape(1);
            default:
                throw new IllegalArgumentException("Invalid parameter name");
        }
    }

    @Override
    public NDArray forward(NDArray data) {
        return forward(new NDList(data)).get(0);
    }

    @Override
    protected NDList opInputs(NDList inputs) {
        return new NDList(inputs.get(0), alpha.getArray());
    }

    @Override
    protected PairList<String, Object> opParams(PairList<String, Object> params) {
        MxOpParams result = new MxOpParams();
        result.addParam("act_type", "prelu");
        result.addAll(params);
        return result;
    }
}
