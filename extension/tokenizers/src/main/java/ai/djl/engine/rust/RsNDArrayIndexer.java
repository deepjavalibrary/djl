/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.engine.rust;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDArrayIndexer;
import ai.djl.ndarray.index.full.NDIndexFullPick;
import ai.djl.ndarray.index.full.NDIndexFullSlice;
import ai.djl.ndarray.index.full.NDIndexFullTake;
import ai.djl.ndarray.types.Shape;

import java.util.Arrays;

/** The {@link NDArrayIndexer} used by the {@link RsNDArray}. */
@SuppressWarnings("try")
public class RsNDArrayIndexer extends NDArrayIndexer {

    private RsNDManager manager;

    RsNDArrayIndexer(RsNDManager manager) {
        this.manager = manager;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullPick fullPick) {
        try (NDScope ignore = new NDScope()) {
            long handle = manager.from(array).getHandle();
            long pickHandle = manager.from(fullPick.getIndices()).getHandle();
            long newHandle = RustLibrary.pick(handle, pickHandle, fullPick.getAxis());
            RsNDArray ret = new RsNDArray(manager, newHandle);
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullTake fullTake) {
        try (NDScope ignore = new NDScope()) {
            long handle = manager.from(array).getHandle();
            long takeHandle = manager.from(fullTake.getIndices()).getHandle();
            RsNDArray ret = new RsNDArray(manager, RustLibrary.take(handle, takeHandle));
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDArray array, NDIndexFullSlice fullSlice) {
        long[] min = fullSlice.getMin();
        long[] max = fullSlice.getMax();
        long[] step = fullSlice.getStep();
        long[] s = array.getShape().getShape().clone();
        if (Arrays.stream(step).anyMatch(i -> i != 1)) {
            throw new UnsupportedOperationException("only step 1 is supported");
        }
        for (int i = 0; i < min.length; i++) {
            if (min[i] >= max[i] || min[i] >= s[i]) {
                Shape shape = fullSlice.getSqueezedShape();
                return manager.create(shape, array.getDataType(), array.getDevice());
            }
        }
        try (NDScope ignore = new NDScope()) {
            long handle = manager.from(array).getHandle();
            long tmp = RustLibrary.fullSlice(handle, min, max, step);
            long newHandle = RustLibrary.reshape(tmp, fullSlice.getSqueezedShape().getShape());
            RustLibrary.deleteTensor(tmp);
            RsNDArray ret = new RsNDArray(manager, newHandle, array.getDataType());
            NDScope.unregister(ret);
            return ret;
        }
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, NDArray value) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDArray array, NDIndexFullSlice fullSlice, Number value) {
        set(array, fullSlice, array.getManager().create(value));
    }
}
