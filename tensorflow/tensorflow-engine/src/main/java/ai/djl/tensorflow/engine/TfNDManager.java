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
package ai.djl.tensorflow.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Path;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.random.RandomStandardNormal;
import org.tensorflow.op.random.RandomUniform;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

public class TfNDManager extends BaseNDManager {

    static final TfNDManager SYSTEM_MANAGER = new SystemManager();

    private static int nameAssignment = 1;
    EagerSession eagerSession;
    Ops tf;
    private static Integer seed;

    private TfNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static TfNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    EagerSession getEagerSession() {
        if (eagerSession == null) {
            eagerSession = EagerSession.options().async(true).build();
        }
        return eagerSession;
    }

    Ops getTf() {
        if (tf == null) {
            tf = Ops.create(eagerSession);
        }
        return tf;
    }

    public static void setRandomSeed(Integer seed) {
        TfNDManager.seed = seed;
    }

    public static Integer getRandomSeed() {
        return seed;
    }

    static int nextNameAssignment() {
        return nameAssignment++;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(byte[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        return new TfNDArray(this, TUint8.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        return new TfNDArray(this, TFloat32.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        return new TfNDArray(this, TInt32.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(boolean[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        return new TfNDArray(this, TBool.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int data) {
        // create scalar tensor with int
        return new TfNDArray(this, TInt32.scalarOf(data));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float data) {
        // create scalar tensor with float
        return new TfNDArray(this, TFloat32.scalarOf(data));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        if (shape.dimension() == 0) {
            // TensorFlow does not support empty scalar(emtpy NDArray with 0 dimension)
            // initialize with scalar 0
            return create(0f).toType(dataType, false);
        }

        Tensor<?> tensor = Tensor.of(TfDataType.toTf(dataType), TfNDArray.toTfShape(shape));
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(Tensor<?> tensor) {
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(ByteBuffer data, Shape shape) {
        try (Tensor<?> tensor =
                Tensor.of(TUint8.DTYPE, TfNDArray.toTfShape(shape), DataBuffers.of(data))) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public TfNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = data.remaining();
        // int8, uint8, boolean use ByteBuffer, so need to explicitly input DataType
        DataType inputType = DataType.fromBuffer(data);

        int numOfBytes = inputType.getNumOfBytes();
        ByteBuffer buf = allocateDirect(size * numOfBytes);

        switch (inputType) {
            case FLOAT32:
                buf.asFloatBuffer().put((FloatBuffer) data);
                break;
            case FLOAT64:
                buf.asDoubleBuffer().put((DoubleBuffer) data);
                break;
            case UINT8:
            case INT8:
            case BOOLEAN:
                buf.put((ByteBuffer) data);
                break;
            case INT32:
                buf.asIntBuffer().put((IntBuffer) data);
                break;
            case INT64:
                buf.asLongBuffer().put((LongBuffer) data);
                break;
            case FLOAT16:
            default:
                throw new AssertionError("Show never happen");
        }
        buf.rewind();

        ByteDataBuffer db = DataBuffers.of(buf);
        Tensor<?> tensor = Tensor.of(TfDataType.toTf(dataType), TfNDArray.toTfShape(shape), db);
        return new TfNDArray(this, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(Buffer data, long[] indptr, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(Buffer data, Shape dataShape, long[] indices, Shape shape) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(TfEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        try (Tensor<?> tensor =
                tf.zeros(tf.constant(shape.getShape()), TfDataType.toTf(dataType)).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        return fill(shape, 1, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        return fill(shape, value, dataType);
    }

    public NDArray fill(Shape shape, Number value, DataType dataType) {
        switch (dataType) {
            case INT32:
                try (Tensor<?> tensor =
                        tf.fill(tf.constant(shape.getShape()), tf.constant(value.intValue()))
                                .asTensor()) {
                    return new TfNDArray(this, tensor);
                }
            case INT64:
                try (Tensor<?> tensor =
                        tf.fill(
                                        tf.constant(shape.getShape()).asOutput(),
                                        tf.constant(value.longValue()))
                                .asTensor()) {
                    return new TfNDArray(this, tensor);
                }
            case FLOAT16:
                try (Tensor<?> tensor =
                        tf.fill(
                                        tf.constant(shape.getShape()).asOutput(),
                                        tf.constant(value.shortValue()))
                                .asTensor()) {
                    return new TfNDArray(this, tensor);
                }
            case FLOAT64:
                try (Tensor<?> tensor =
                        tf.fill(
                                        tf.constant(shape.getShape()).asOutput(),
                                        tf.constant(value.doubleValue()))
                                .asTensor()) {
                    return new TfNDArray(this, tensor);
                }
            default:
                try (Tensor<?> tensor =
                        tf.fill(
                                        tf.constant(shape.getShape()).asOutput(),
                                        tf.constant(value.floatValue()))
                                .asTensor()) {
                    return new TfNDArray(this, tensor);
                }
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        if (stop <= start && step > 0) {
            return create(new Shape(0), dataType);
        }
        try (Tensor<?> tensor =
                tf.range(
                                toConstant(start, dataType),
                                toConstant(stop, dataType),
                                toConstant(step, dataType))
                        .asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        return eyeHelper(rows, cols, k, dataType);
    }

    private <T extends TType> NDArray eyeHelper(int rows, int cols, int k, DataType dataType) {
        Operand<T> diagonal =
                ((TfNDArray) ones(new Shape(Math.min(rows, cols)), dataType)).getOperand();

        Operand<T> value = ((TfNDArray) zeros(new Shape(1))).getOperand();
        Operand<T> output =
                tf.linalg.matrixDiag(
                        diagonal, tf.constant(k), tf.constant(rows), tf.constant(cols), value);
        try (Tensor<?> tensor = output.asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    <T extends TType> Constant<T> toConstant(Number n, DataType jType) {
        return TfNDArray.getConstant(n, jType, tf);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint) {
        if (num < 0) {
            throw new IllegalArgumentException("number of samples must be non-negative.");
        }
        if (num == 0) {
            return create(new Shape(0));
        }
        if (endpoint) {

            try (Tensor<?> tensor =
                    org.tensorflow.op.core.LinSpace.create(
                                    tf.scope(),
                                    tf.constant(start),
                                    tf.constant(stop),
                                    tf.constant(num))
                            .asTensor()) {
                return new TfNDArray(this, tensor);
            }
        }
        try (Tensor<?> tensor =
                org.tensorflow.op.core.LinSpace.create(
                                tf.scope(),
                                tf.constant(start),
                                tf.constant(stop),
                                tf.constant(num + 1))
                        .asTensor()) {
            return new TfNDArray(this, tensor).get(new NDIndex(":-1"));
        }
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        Operand shapeOp = tf.constant(shape.getShape());
        org.tensorflow.DataType dType;
        if (dataType == DataType.UNKNOWN) {
            dType = TFloat32.DTYPE;
        } else {
            dType = TfDataType.toTf(dataType);
        }
        Operand minVal = tf.dtypes.cast(tf.constant(low), dType);
        Operand maxVal = tf.dtypes.cast(tf.constant(high), dType);
        Operand result;
        if (seed != null) {
            result =
                    tf.random.randomUniform(
                            shapeOp,
                            dType,
                            RandomUniform.seed((long) 1234),
                            RandomUniform.seed2((long) 2234));
        } else {
            result = tf.random.randomUniform(shapeOp, dType);
        }
        try (Tensor<?> tensor =
                tf.math.add(tf.math.mul(result, tf.math.sub(maxVal, minVal)), minVal).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        Operand shapeOp = tf.dtypes.cast(tf.constant(shape.getShape()), TInt32.DTYPE);
        org.tensorflow.DataType dType;
        if (dataType == DataType.UNKNOWN) {
            dType = TFloat32.DTYPE;
        } else {
            dType = TfDataType.toTf(dataType);
        }
        Operand mean = tf.dtypes.cast(tf.constant(loc), dType);
        Operand std = tf.dtypes.cast(tf.constant(scale), dType);
        Operand result;
        if (seed != null) {
            result =
                    tf.random.randomStandardNormal(
                            shapeOp,
                            dType,
                            RandomStandardNormal.seed((long) 1234),
                            RandomStandardNormal.seed2((long) 2234));
        } else {
            result = tf.random.randomStandardNormal(shapeOp, dType);
        }
        try (Tensor<?> tensor = tf.math.add(tf.math.mul(result, std), mean).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public TfNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public TfNDManager newSubManager(Device device) {
        TfNDManager manager = new TfNDManager(this, device);
        attach(manager.uid, manager);
        // initialize eager sessions and operators only for sub managers
        manager.getEagerSession();
        manager.getTf();
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isOpen() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public void detach(String resourceId) {
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (eagerSession != null) {
            eagerSession.close();
        }
    }

    private static final class SystemManager extends TfNDManager {

        SystemManager() {
            super(null, Device.defaultDevice());
        }

        /** {@inheritDoc} */
        @Override
        public void attach(String resrouceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
