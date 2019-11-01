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

import ai.djl.MalformedModelException;
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

/**
 * Applies Leaky Parametric ReLU activation element-wise to the input.
 *
 * <p>Leaky ReLUs attempt to fix the 'dying ReLU' problem by allowing a small slope when the input
 * is negative and has a slope of one when input is positive. This is defined by \(y= x \gt 0 ? x :
 * slope * x\).
 *
 * <p>Parametric ReLU is a Leaky ReLU in which the slope is learnt during training.
 */
public class Prelu extends ParameterBlock {

    private static final byte VERSION = 1;

    private Parameter alpha;

    /** Creates a Parametric ReLU Block. */
    public Prelu() {
        alpha = new Parameter("alpha", this, ParameterType.OTHER);
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        NDArray data = inputs.singletonOrThrow();

        NDList list = new NDList(data, parameterStore.getValue(alpha, data.getDevice()));
        NDArrayEx ex = data.getNDArrayInternal();
        return ex.prelu(list, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputs) {
        return new Shape[] {inputs[0]};
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.singletonList(alpha);
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        if ("alpha".equals(name)) {
            return new Shape();
        }
        throw new IllegalArgumentException("Invalid parameter name");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        alpha.save(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        alpha.load(manager, is);
    }
}
