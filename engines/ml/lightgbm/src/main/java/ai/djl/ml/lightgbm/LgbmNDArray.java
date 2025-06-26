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
package ai.djl.ml.lightgbm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;

import com.microsoft.ml.lightgbm.SWIGTYPE_p_double;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_float;
import com.microsoft.ml.lightgbm.SWIGTYPE_p_void;
import com.microsoft.ml.lightgbm.lightgbmlib;
import com.microsoft.ml.lightgbm.lightgbmlibConstants;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicReference;

/** {@code LgbmNDArray} is the LightGBM implementation of {@link NDArray}. */
public class LgbmNDArray extends NDArrayAdapter {

    private ByteBuffer data;
    private SparseFormat format;

    private AtomicReference<SWIGTYPE_p_void> handle;
    private int typeConstant;
    private AtomicReference<SWIGTYPE_p_float> floatDataRef;
    private AtomicReference<SWIGTYPE_p_double> doubleDataRef;

    LgbmNDArray(
            NDManager manager,
            NDManager alternativeManager,
            ByteBuffer data,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, NDManager.nextUid());
        this.data = data;
        this.format = SparseFormat.DENSE;
        manager.attachInternal(uid, this);
        handle = new AtomicReference<>();
        floatDataRef = new AtomicReference<>();
        doubleDataRef = new AtomicReference<>();
    }

    /** {@inheritDoc} */
    @Override
    public void returnResource(NDManager manager) {
        detach();
        this.manager = manager;
        manager.attachUncappedInternal(getUid(), this);
    }

    /**
     * Returns the native LightGBM handle to the array.
     *
     * @return the native LightGBM handle to the array
     */
    public SWIGTYPE_p_void getHandle() {
        if (handle.get() == null) {
            if (shape.dimension() != 2) {
                throw new IllegalArgumentException(
                        "The LightGBM operation can only be performed with a 2-dimensional matrix,"
                                + " but was passed an NDArray with "
                                + shape.dimension()
                                + " dimensions");
            }
            int size = Math.toIntExact(size());

            if (getDataType() == DataType.FLOAT32) {
                typeConstant = lightgbmlibConstants.C_API_DTYPE_FLOAT32;
                FloatBuffer d1 = toByteBuffer().asFloatBuffer();
                floatDataRef.set(lightgbmlib.new_floatArray(size));
                for (int i = 0; i < size; i++) {
                    lightgbmlib.floatArray_setitem(floatDataRef.get(), i, d1.get(i));
                }
                handle.set(lightgbmlib.float_to_voidp_ptr(floatDataRef.get()));
            } else if (getDataType() == DataType.FLOAT64) {
                typeConstant = lightgbmlibConstants.C_API_DTYPE_FLOAT64;
                DoubleBuffer d1 = toByteBuffer().asDoubleBuffer();
                doubleDataRef.set(lightgbmlib.new_doubleArray(size));
                for (int i = 0; i < size; i++) {
                    lightgbmlib.doubleArray_setitem(doubleDataRef.get(), i, d1.get(i));
                }
                handle.set(lightgbmlib.double_to_voidp_ptr(doubleDataRef.get()));
            } else {
                throw new IllegalArgumentException(
                        "The LightGBM operation can only be performed with a Float32 or Float64"
                                + " array, but was given a "
                                + getDataType());
            }
        }
        return handle.get();
    }

    /**
     * Returns the number of data rows (assuming a 2D matrix).
     *
     * @return the number of data rows (assuming a 2D matrix)
     */
    public int getRows() {
        return Math.toIntExact(shape.get(0));
    }

    /**
     * Returns the number of data cols (assuming a 2D matrix).
     *
     * @return the number of data cols (assuming a 2D matrix)
     */
    public int getCols() {
        return Math.toIntExact(shape.get(1));
    }

    /**
     * Returns the LightGBM type constant of the array.
     *
     * @return the LightGBM type constant of the array
     */
    public int getTypeConstant() {
        return typeConstant;
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return format;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer(boolean tryDirect) {
        if (data == null) {
            throw new UnsupportedOperationException("Cannot obtain value from DMatrix");
        }
        data.rewind();
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        LgbmNDArray array = (LgbmNDArray) replaced;

        SWIGTYPE_p_float floatData = floatDataRef.getAndSet(array.floatDataRef.getAndSet(null));
        if (floatData != null) {
            lightgbmlib.delete_floatArray(floatData);
        }
        SWIGTYPE_p_double doubleData = doubleDataRef.getAndSet(array.doubleDataRef.getAndSet(null));
        if (doubleData != null) {
            lightgbmlib.delete_doubleArray(doubleData);
        }
        handle.set(array.handle.getAndSet(null));
        data = array.data;
        format = array.format;
        typeConstant = array.typeConstant;
        shape = array.shape;
        dataType = array.dataType;
        replaced.close();
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = LgbmNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        SWIGTYPE_p_float floatData = floatDataRef.getAndSet(null);
        if (floatData != null) {
            lightgbmlib.delete_floatArray(floatData);
        }
        SWIGTYPE_p_double doubleData = doubleDataRef.getAndSet(null);
        if (doubleData != null) {
            lightgbmlib.delete_doubleArray(doubleData);
        }
        handle.set(null);
        data = null;
    }
}
