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
import org.tensorflow.EagerSession;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import org.tensorflow.op.Ops;
import org.tensorflow.op.image.DecodeJpeg;

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
        return new TfNDArray(
                this, tf.image.decodeJpeg(tf.constant(data), DecodeJpeg.channels((long) 3)));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data, Shape shape) {
        return new TfNDArray(this, Tensors.create(data));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int data) {
        return new TfNDArray(this, Tensors.create(data));
    }

    public TfNDArray create(Tensor<?> tensor) {
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(ByteBuffer data, Shape shape) {
        return new TfNDArray(this, shape, data);
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
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    @Override
    public Engine getEngine() {
        return Engine.getEngine(TfEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        switch (dataType) {
            case INT32:
                return new TfNDArray(this, tf.zeros(tf.constant(shape.getShape()), Integer.class));
            case INT64:
                return new TfNDArray(this, tf.zeros(tf.constant(shape.getShape()), Long.class));
            case FLOAT16:
                return new TfNDArray(this, tf.zeros(tf.constant(shape.getShape()), Short.class));
            case FLOAT64:
                return new TfNDArray(this, tf.zeros(tf.constant(shape.getShape()), Double.class));
            default:
                return new TfNDArray(this, tf.zeros(tf.constant(shape.getShape()), Float.class));
        }
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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return null;
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
