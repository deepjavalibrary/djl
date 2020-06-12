/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorflow.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.index.dim.NDIndexBooleans;
import ai.djl.ndarray.index.full.NDIndexFullSlice;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Squeeze;
import org.tensorflow.types.TInt64;

/** The {@link NDArrayIndexer} used by the {@link TfNDArray}. */
public class TfNDArrayIndexer extends NDArrayIndexer {

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        TfNDArray tfArray = (TfNDArray) array;
        Ops tf = ((TfNDManager) tfArray.getManager()).getTf();
        Constant<TInt64> begin = tf.constant(fullSlice.getMin());
        Constant<TInt64> end = tf.constant(fullSlice.getMax());
        Constant<TInt64> step = tf.constant(fullSlice.getStep());
        int[] toSqueeze = fullSlice.getToSqueeze();
        Operand<?> sliced = tf.stridedSlice(tfArray.asOperand(), begin, end, step);
        if (toSqueeze.length > 0) {
            List<Long> squeeze =
                    Arrays.stream(toSqueeze).mapToLong(i -> i).boxed().collect(Collectors.toList());
            sliced = tf.squeeze(sliced, Squeeze.axis(squeeze));
        }
        return new TfNDArray(array.getManager(), sliced);
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexBooleans indices, NDArray value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }
}
