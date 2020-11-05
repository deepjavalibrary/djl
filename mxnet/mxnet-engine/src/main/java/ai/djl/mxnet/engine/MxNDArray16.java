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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.sun.jna.Pointer;

/** MXNet 1.6.0 {@code MxNDArray} implementation. */
public class MxNDArray16 extends MxNDArray {

    MxNDArray16(
            MxNDManager manager,
            Pointer handle,
            Device device,
            Shape shape,
            DataType dataType,
            boolean hasGradient) {
        super(manager, handle, device, shape, dataType, hasGradient);
    }

    MxNDArray16(MxNDManager manager, Pointer handle) {
        super(manager, handle);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return manager.invoke("_np_zeros_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return manager.invoke("_np_ones_like", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        // be careful that MXNet numpy argsort op didn't officially support this param
        params.addParam("is_ascend", ascending);
        params.setDataType(DataType.INT32);
        return manager.invoke("argsort", this, params).toType(DataType.INT64, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        if (isEmpty() || isScalar()) {
            long dim = getShape().dimension();
            if (axis >= dim) {
                throw new IllegalArgumentException(
                        "axis " + axis + "is out of bounds for array of dimension " + dim);
            }
            return duplicate();
        }

        MxOpParams params = new MxOpParams();
        params.addParam("axis", axis);
        return manager.invoke("sort", this, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        if (isEmpty() || isScalar()) {
            return duplicate();
        }
        return manager.invoke("sort", this, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        try (NDArray array = super.argMax()) {
            return array.toType(DataType.INT64, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        try (NDArray array = super.argMax(axis)) {
            return array.toType(DataType.INT64, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        try (NDArray array = super.argMin()) {
            return array.toType(DataType.INT64, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        try (NDArray array = super.argMin(axis)) {
            return array.toType(DataType.INT64, true);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        if (isScalar()) {
            return reshape(1);
        }
        return super.expandDims(axis);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray matMul(NDArray other) {
        throw new UnsupportedOperationException("matMul is not supported in MXNet 1.6.0");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return manager.invoke("_npi_not_equal", new NDArray[] {this, this}, null);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        MxOpParams params = new MxOpParams();
        params.setShape(shape);
        return manager.invoke("_np_broadcast_to", this, params);
    }
}
