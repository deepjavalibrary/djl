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
package ai.djl.mxnet.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.index.full.NDIndexFullPick;
import ai.djl.ndarray.index.full.NDIndexFullSlice;
import ai.djl.ndarray.index.full.NDIndexFullTake;
import ai.djl.ndarray.types.Shape;

import java.util.Stack;

/** The {@link NDArrayIndexer} used by the {@link MxNDArray}. */
public class MxNDArrayIndexer extends NDArrayIndexer {

    private MxNDManager manager;

    MxNDArrayIndexer(MxNDManager manager) {
        this.manager = manager;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullPick fullPick) {
        array = manager.from(array);
        MxOpParams params = new MxOpParams();
        params.addParam("axis", fullPick.getAxis());
        params.addParam("keepdims", true);
        params.add("mode", "wrap");
        NDList pick = new NDList(array, manager.from(fullPick.getIndices()));
        return manager.invoke("pick", pick, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullTake fullTake) {
        array = manager.from(array);
        MxOpParams params = new MxOpParams();
        params.addParam("axis", fullTake.getAxis());
        params.add("mode", "wrap");
        NDList pick = new NDList(array, manager.from(fullTake.getIndices()));
        return manager.invoke("take", pick, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        array = manager.from(array);
        long[] min = fullSlice.getMin();
        long[] max = fullSlice.getMax();
        long[] s = array.getShape().getShape();
        for (int i = 0; i < min.length; i++) {
            if (min[i] >= max[i] || min[i] >= s[i]) {
                return manager.create(new Shape(0));
            }
        }

        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        NDArray result = manager.invoke("_npi_slice", array, params);
        int[] toSqueeze = fullSlice.getToSqueeze();
        if (toSqueeze.length > 0) {
            NDArray oldResult = result;
            result = result.squeeze(toSqueeze);
            oldResult.close();
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        array = manager.from(array);
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());

        Stack<NDArray> prepareValue = new Stack<>();
        prepareValue.add(value);
        prepareValue.add(prepareValue.peek().toDevice(array.getDevice(), false));
        // prepareValue.add(prepareValue.peek().asType(getDataType(), false));
        // Deal with the case target: (1, 10, 1), original (10)
        // try to find (10, 1) and reshape (10) to that
        Shape targetShape = fullSlice.getShape();
        while (targetShape.size() > value.size()) {
            targetShape = targetShape.slice(1);
        }
        prepareValue.add(prepareValue.peek().reshape(targetShape));
        prepareValue.add(prepareValue.peek().broadcast(fullSlice.getShape()));

        manager.invoke(
                "_npi_slice_assign",
                new NDArray[] {array, prepareValue.peek()},
                new NDArray[] {array},
                params);
        for (NDArray toClean : prepareValue) {
            if (toClean != value) {
                toClean.close();
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        array = manager.from(array);
        MxOpParams params = new MxOpParams();
        params.addTupleParam("begin", fullSlice.getMin());
        params.addTupleParam("end", fullSlice.getMax());
        params.addTupleParam("step", fullSlice.getStep());
        params.addParam("scalar", value);
        manager.invoke(
                "_npi_slice_assign_scalar", new NDArray[] {array}, new NDArray[] {array}, params);
    }
}
