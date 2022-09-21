/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.stream.IntStream;

/**
 * {@code SparseMax} contains a generic implementation of sparsemax function the definition of
 * SparseMax can be referred to https://arxiv.org/pdf/1602.02068.pdf. {@code SparseMax} is a simpler
 * implementation of sparseMax function, where we set K as a hyperParameter(default 3). We only do
 * softmax on those max-K data, and we set all the other value as 0.
 */
public class SparseMax extends AbstractBlock {
    private static final Byte VERSION = 1;

    private int axis;
    private int topK;

    /** Creates a sparseMax activation function for the last axis and 3 elements. */
    public SparseMax() {
        this(-1, 3);
    }

    /**
     * Creates a sparseMax activation function along a given axis for 3 elements.
     *
     * @param axis the axis to do sparseMax for
     */
    public SparseMax(int axis) {
        this(axis, 3);
    }

    /**
     * Creates a sparseMax activation function along a given axis and number of elements.
     *
     * @param axis the axis to do sparseMax for
     * @param topK hyperParameter K
     */
    public SparseMax(int axis, int topK) {
        super(VERSION);
        this.axis = axis;
        this.topK = topK;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // the shape of input and output are the same
        return new Shape[0];
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        /*
        A simple implementation of sparseMax, where we only calculate softMax with largest K data
         */
        NDArray input = inputs.singletonOrThrow();
        if (axis != -1) {
            input = input.swapAxes(axis, -1);
        }

        // level should be: the max i-th is index j in input
        NDArray level = input.argSort(-1, false).toType(DataType.INT64, false);
        int lastDimSize = (int) input.size(input.getShape().dimension() - 1);

        // maskTopK should be: the topK in input is 1 and other is zero
        NDArray maskTopK =
                NDArrays.add(
                        IntStream.range(0, topK)
                                .mapToObj(j -> level.get("..., {}", j).oneHot(lastDimSize))
                                .toArray(NDArray[]::new));

        NDArray expSum =
                input.exp().mul(maskTopK).sum(new int[] {-1}, true).broadcast(input.getShape());
        NDArray output = input.exp().mul(maskTopK).div(expSum);

        if (axis != -1) {
            output = output.swapAxes(axis, -1);
        }
        return new NDList(output);
    }
}
