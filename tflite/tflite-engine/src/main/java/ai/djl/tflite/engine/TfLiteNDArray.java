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
package ai.djl.tflite.engine;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrayAdapter;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.UUID;
import org.tensorflow.lite.Tensor;

/** {@code TfLiteNDArray} is the TFLite implementation of {@link NDArray}. */
public class TfLiteNDArray implements NDArrayAdapter {

    private TfLiteNDManager manager;
    private Tensor tensor;
    private ByteBuffer data;
    private Shape shape;
    private DataType dataType;
    private String name;
    private boolean isClosed;
    private String uid;

    TfLiteNDArray(TfLiteNDManager manager, Tensor tensor) {
        this.manager = manager;
        uid = UUID.randomUUID().toString();
        manager.attachInternal(uid, this);
        this.tensor = tensor;
        shape = new Shape(Arrays.stream(tensor.shape()).mapToLong(i -> i).toArray());
        dataType = TfLiteDataType.fromTf(tensor.dataType());
    }

    TfLiteNDArray(TfLiteNDManager manager, ByteBuffer data, Shape shape, DataType dataType) {
        this.manager = manager;
        this.data = data;
        this.shape = shape;
        this.dataType = dataType;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return name;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {
        this.name = name;
    }

    /** {@inheritDoc} */
    @Override
    public String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return dataType;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        // TODO: Support on multiple devices
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        return shape;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(NDManager manager) {
        detach();
        this.manager = (TfLiteNDManager) manager;
        manager.attachInternal(getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void tempAttach(NDManager manager) {
        detach();
        NDManager original = this.manager;
        this.manager = (TfLiteNDManager) manager;
        manager.tempAttachInternal(original, getUid(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void detach() {
        manager.detachInternal(getUid());
        manager = TfLiteNDManager.getSystemManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toType(DataType dataType, boolean copy) {
        if (dataType.equals(this.dataType)) {
            if (copy) {
                return new TfLiteNDArray(manager, toByteBuffer().duplicate(), shape, dataType);
            } else {
                return this;
            }
        }
        Number[] array = toArray();
        switch (dataType) {
            case FLOAT64:
                double[] doubleResult =
                        Arrays.stream(array).mapToDouble(Number::doubleValue).toArray();
                return manager.create(doubleResult).reshape(shape);
            case FLOAT32:
                float[] floatResult = new float[array.length];
                for (int i = 0; i < array.length; i++) {
                    floatResult[i] = array[i].floatValue();
                }
                return manager.create(floatResult).reshape(shape);
            case INT32:
                int[] intResult = Arrays.stream(array).mapToInt(Number::intValue).toArray();
                return manager.create(intResult).reshape(shape);
            case INT64:
                long[] longResult = Arrays.stream(array).mapToLong(Number::longValue).toArray();
                return manager.create(longResult).reshape(shape);
            case INT8:
                byte[] booleanResult = new byte[array.length];
                for (int i = 0; i < array.length; i++) {
                    booleanResult[i] = array[i].byteValue();
                }
                return manager.create(booleanResult).reshape(shape);
            default:
                throw new UnsupportedOperationException(
                        "Type conversion is not supported for TFLite for data type " + dataType);
        }
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        if (data == null) {
            data = tensor.buffer();
        }
        data.rewind();
        return data;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        if (tensor != null) {
            throw new UnsupportedOperationException("Not supported for TFLite");
        } else {
            if (Arrays.stream(shape.getShape()).anyMatch(n -> n < 0)) {
                throw new UnsupportedOperationException(
                        "Negative shape is not supported for TFLite");
            }
            return new TfLiteNDArray(manager, data, shape, dataType);
        }
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (isClosed) {
            return "This array is already closed";
        }
        return toDebugString();
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        isClosed = true;
    }
}
