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
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.IOException;

/**
 * Applies Leaky Parametric ReLU activation element-wise to the input.
 *
 * <p>Leaky ReLUs attempt to fix the 'dying ReLU' problem by allowing a small slope when the input
 * is negative and has a slope of one when input is positive. This is defined by \(y= x \gt 0 ? x :
 * slope * x\).
 *
 * <p>Parametric ReLU is a Leaky ReLU in which the slope is learnt during training.
 */
public class Prelu extends AbstractBlock {

    private static final byte VERSION = 2;

    private Parameter alpha;

    /** Creates a Parametric ReLU Block. */
    public Prelu() {
        super(VERSION);
        alpha =
                addParameter(
                        Parameter.builder()
                                .setName("alpha")
                                .setType(Parameter.Type.WEIGHT)
                                .optShape(new Shape())
                                .build());
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        NDArray alphaArr = parameterStore.getValue(alpha, input.getDevice(), training);
        return prelu(input, alphaArr);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return new Shape[] {inputs[0]};
    }

    /** {@inheritDoc} */
    @Override
    public void loadMetadata(byte loadVersion, DataInputStream is)
            throws IOException, MalformedModelException {
        if (loadVersion == version) {
            readInputShapes(is);
        } else if (loadVersion != 1) {
            throw new MalformedModelException("Unsupported encoding version: " + loadVersion);
        }
    }

    /**
     * Applies a Prelu activation on the input {@link NDArray}.
     *
     * <p>Prelu is defined as \(y = max(0,x) + alpha * min(0, x) \) where alpha is learnable
     * parameter
     *
     * @param input the input {@link NDArray}
     * @param alpha learnable parameter
     * @return the {@link NDArray} after applying Prelu activation
     */
    public static NDList prelu(NDArray input, NDArray alpha) {
        NDArrayEx ex = input.getNDArrayInternal();
        return ex.prelu(input, alpha);
    }
}
