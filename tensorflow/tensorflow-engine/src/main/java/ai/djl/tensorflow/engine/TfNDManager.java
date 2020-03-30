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
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.tools.buffer.ByteDataBuffer;
import org.tensorflow.tools.buffer.DataBuffers;
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

    static int nextNameAssignment() {
        return nameAssignment++;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(byte[] data) {
        org.tensorflow.tools.Shape shape = org.tensorflow.tools.Shape.of(data.length);
        return new TfNDArray(this, TUint8.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data) {
        org.tensorflow.tools.Shape shape = org.tensorflow.tools.Shape.of(data.length);
        return new TfNDArray(this, TFloat32.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int[] data) {
        org.tensorflow.tools.Shape shape = org.tensorflow.tools.Shape.of(data.length);
        return new TfNDArray(this, TInt32.tensorOf(shape, DataBuffers.of(data)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(boolean[] data) {
        org.tensorflow.tools.Shape shape = org.tensorflow.tools.Shape.of(data.length);
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
    public NDArray create(Shape shape, DataType dataType, Device device) {
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
        return new TfNDArray(this, shape, data);
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
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(TfEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return new TfNDArray(
                this, tf.zeros(tf.constant(shape.getShape()), TfDataType.toTf(dataType)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return fill(shape, 1, dataType, device);
    }

    public NDArray fill(Shape shape, Number value, DataType dataType, Device device) {
        switch (dataType) {
            case INT32:
                return new TfNDArray(
                        this,
                        tf.fill(tf.constant(shape.getShape()), tf.constant(value.intValue())));
            case INT64:
                return new TfNDArray(
                        this,
                        tf.fill(
                                tf.constant(shape.getShape()).asOutput(),
                                tf.constant(value.longValue())));
            case FLOAT16:
                return new TfNDArray(
                        this,
                        tf.fill(
                                tf.constant(shape.getShape()).asOutput(),
                                tf.constant(value.shortValue())));
            case FLOAT64:
                return new TfNDArray(
                        this,
                        tf.fill(
                                tf.constant(shape.getShape()).asOutput(),
                                tf.constant(value.doubleValue())));
            default:
                return new TfNDArray(
                        this,
                        tf.fill(
                                tf.constant(shape.getShape()).asOutput(),
                                tf.constant(value.floatValue())));
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType, Device device) {
        return new TfNDArray(
                this,
                tf.range(
                        toConstant(start, dataType),
                        toConstant(stop, dataType),
                        toConstant(step, dataType)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return eyeHelper(rows, cols, k, dataType, device);
    }

    private <T extends TType> NDArray eyeHelper(
            int rows, int cols, int k, DataType dataType, Device device) {
        Operand<T> diagonal =
                ((TfNDArray) ones(new Shape(Math.min(rows, cols)), dataType, device)).asOperand();

        Operand<T> value = ((TfNDArray) zeros(new Shape(1))).asOperand();
        Operand<T> output =
                tf.linalg.matrixDiag(
                        diagonal, tf.constant(k), tf.constant(rows), tf.constant(cols), value);
        return new TfNDArray(this, output);
    }

    <T extends TType> Constant<T> toConstant(Number n, DataType jType) {
        return TfNDArray.getConstant(n, jType, tf);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(float start, float stop, int num, boolean endpoint, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            float low, float high, Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            float loc, float scale, Shape shape, DataType dataType, Device device) {
        return null;
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
        eagerSession.close();
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
