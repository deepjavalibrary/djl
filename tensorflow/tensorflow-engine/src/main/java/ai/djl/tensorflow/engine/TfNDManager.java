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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import org.tensorflow.EagerSession;
import org.tensorflow.OperationBuilder;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

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
        try (Tensor tensor = TUint8.tensorOf(shape, DataBuffers.of(data))) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        try (Tensor tensor = TFloat32.tensorOf(shape, DataBuffers.of(data))) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        try (Tensor tensor = TInt32.tensorOf(shape, DataBuffers.of(data))) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(boolean[] data) {
        org.tensorflow.ndarray.Shape shape = org.tensorflow.ndarray.Shape.of(data.length);
        try (Tensor tensor = TBool.tensorOf(shape, DataBuffers.of(data))) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int data) {
        // create scalar tensor with int
        try (Tensor tensor = TInt32.scalarOf(data)) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float data) {
        // create scalar tensor with float
        try (Tensor tensor = TFloat32.scalarOf(data)) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String data) {
        try (Tensor tensor = TString.scalarOf(data)) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(String[] data) {
        try (Tensor tensor = TString.vectorOf(data)) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType) {
        if (shape.dimension() == 0) {
            // TensorFlow does not support empty scalar(emtpy NDArray with 0 dimension)
            // initialize with scalar 0
            return create(0f).toType(dataType, false);
        }

        Tensor tensor = Tensor.of(TfDataType.toClassType(dataType), TfNDArray.toTfShape(shape));
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(Tensor tensor) {
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(ByteBuffer data, Shape shape) {
        try (Tensor tensor =
                Tensor.of(TUint8.class, TfNDArray.toTfShape(shape), DataBuffers.of(data))) {
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
        try (Tensor tensor =
                Tensor.of(TfDataType.toClassType(dataType), TfNDArray.toTfShape(shape), db)) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(TfEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType) {
        return full(shape, 0, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType) {
        return full(shape, 1, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray full(Shape shape, float value, DataType dataType) {
        TfNDArray valueArr = (TfNDArray) create(value).toType(dataType, false);
        TfNDArray dimArr = (TfNDArray) create(shape.getShape());
        OperationBuilder opBuilder = getEagerSession().opBuilder("Fill", "Fill");
        opBuilder.addInput(dimArr.getHandle().asOutput());
        opBuilder.addInput(valueArr.getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(float start, float stop, float step, DataType dataType) {
        if (stop <= start && step > 0) {
            return create(new Shape(0), dataType);
        }
        OperationBuilder opBuilder = getEagerSession().opBuilder("Range", "Range");
        opBuilder.addInput(
                ((TfNDArray) create(start).toType(dataType, false)).getHandle().asOutput());
        opBuilder.addInput(
                ((TfNDArray) create(stop).toType(dataType, false)).getHandle().asOutput());
        opBuilder.addInput(
                ((TfNDArray) create(step).toType(dataType, false)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType) {
        OperationBuilder opBuilder = getEagerSession().opBuilder("MatrixDiagV2", "MatrixDiag");
        opBuilder.addInput(
                ((TfNDArray) ones(new Shape(Math.min(rows, cols)), dataType))
                        .getHandle()
                        .asOutput());
        opBuilder.addInput(((TfNDArray) create(k)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) create(rows)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) create(cols)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) zeros(new Shape(1))).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(this, tensor);
        }
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
        if (!endpoint && num > 1) {
            stop -= (int) ((stop - start) / num);
        }
        OperationBuilder opBuilder = getEagerSession().opBuilder("LinSpace", "LinSpace");
        opBuilder.addInput(((TfNDArray) create(start)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) create(stop)).getHandle().asOutput());
        opBuilder.addInput(((TfNDArray) create(num)).getHandle().asOutput());
        try (Tensor tensor = opBuilder.build().output(0).asTensor()) {
            return new TfNDArray(this, tensor);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(float low, float high, Shape shape, DataType dataType) {
        if (DataType.STRING.equals(dataType)) {
            throw new IllegalArgumentException("String data type is not supported!");
        }
        OperationBuilder opBuilder = getEagerSession().opBuilder("RandomUniform", "RandomUniform");
        opBuilder.addInput(((TfNDArray) create(shape.getShape())).getHandle().asOutput());
        opBuilder.setAttr("dtype", TfDataType.toProtoType(dataType));
        if (seed != null) {
            opBuilder.setAttr("seed", 1234);
            opBuilder.setAttr("seed2", 1234);
        }
        try (Tensor tensor = opBuilder.build().output(0).asTensor();
                NDArray temp1 = new TfNDArray(this, tensor);
                NDArray temp2 = temp1.mul(high - low)) {
            return temp2.add(low);
        }
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(float loc, float scale, Shape shape, DataType dataType) {
        if (DataType.STRING.equals(dataType)) {
            throw new IllegalArgumentException("String data type is not supported!");
        }
        OperationBuilder opBuilder =
                getEagerSession().opBuilder("RandomStandardNormal", "RandomStandardNormal");
        opBuilder.addInput(((TfNDArray) create(shape.getShape())).getHandle().asOutput());
        opBuilder.setAttr("dtype", TfDataType.toProtoType(dataType));
        if (seed != null) {
            opBuilder.setAttr("seed", 1234);
            opBuilder.setAttr("seed2", 1234);
        }
        try (Tensor tensor = opBuilder.build().output(0).asTensor();
                NDArray temp1 = new TfNDArray(this, tensor);
                NDArray temp2 = temp1.mul(scale)) {
            return temp2.add(loc);
        }
    }

    /** {@inheritDoc} */
    @Override
    public TfNDManager newSubManager(Device device) {
        TfNDManager manager = new TfNDManager(this, device);
        attachInternal(manager.uid, manager);
        // initialize eager sessions and operators only for sub managers
        manager.getEagerSession();
        manager.getTf();
        return manager;
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
            super(null, null);
        }

        /** {@inheritDoc} */
        @Override
        public void attachInternal(String resrouceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detachInternal(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
