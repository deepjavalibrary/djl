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
package ai.djl.nn.core;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterBlock;
import ai.djl.nn.ParameterType;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

public class Prelu extends ParameterBlock {

    private static final byte VERSION = 1;

    private Parameter alpha;

    public Prelu() {
        alpha = new Parameter("alpha", this, ParameterType.OTHER);
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDArray data = inputs.head();

        NDList list = new NDList(data, parameterStore.getValue(alpha, data.getDevice()));
        NDArrayEx ex = data.getNDArrayInternal();
        return ex.prelu(list, params);
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        return new Shape[] {inputs[0]};
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(alpha);
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        if ("alpha".equals(name)) {
            return new Shape();
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        alpha.save(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is) throws IOException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new IllegalArgumentException("Unsupported encoding version: " + version);
        }
        alpha.load(manager, is);
    }
}
