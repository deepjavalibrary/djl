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
public class TfLiteNDArray extends NDArrayAdapter {

    private Tensor tensor;
    private ByteBuffer data;

    TfLiteNDArray(NDManager manager, NDManager alternativeManager, Tensor tensor) {
        super(
                manager,
                alternativeManager,
                new Shape(Arrays.stream(tensor.shape()).mapToLong(i -> i).toArray()),
                TfLiteDataType.fromTf(tensor.dataType()),
                UUID.randomUUID().toString());
        this.tensor = tensor;
        manager.attachInternal(uid, this);
    }

    TfLiteNDArray(
            NDManager manager,
            NDManager alternativeManager,
            ByteBuffer data,
            Shape shape,
            DataType dataType) {
        super(manager, alternativeManager, shape, dataType, UUID.randomUUID().toString());
        this.data = data;
        manager.attachInternal(uid, this);
    }

    /** {@inheritDoc} */
    @Override
    public void intern(NDArray replaced) {
        if (tensor != null) {
            tensor.close();
        }
        this.data = ((TfLiteNDArray) replaced).data;
        this.tensor = ((TfLiteNDArray) replaced).tensor;
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
            if (!copy) {
                return this;
            }
            return new TfLiteNDArray(
                    manager, alternativeManager, toByteBuffer().duplicate(), shape, dataType);
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
            return new TfLiteNDArray(manager, alternativeManager, data, shape, dataType);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (tensor != null) {
            tensor.close();
        }
    }
}
