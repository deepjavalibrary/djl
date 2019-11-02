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
package ai.djl.mxnet.engine;

import ai.djl.Device;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.PairList;
import com.sun.jna.Pointer;
import java.lang.ref.WeakReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

public class MxNDManager implements NDManager {

    /**
     * A global {@link NDManager} singleton instance.
     *
     * <p>This NDManager is the root of all the other {@code NDManager}s. NDArrays created by this
     * manager are un-managed, user has to close them manually. Those NDArrays will be released on
     * GC, and might be run into out of native memory issue.
     */
    private static final MxNDManager SYSTEM_MANAGER = new SystemManager();

    private static final NDList EMPTY_LIST = new NDList(0);
    private static final NDArray[] EMPTY_ARRAY = new NDArray[0];

    private NDManager parent;
    private String uid;
    private Device device;
    private Map<String, WeakReference<AutoCloseable>> resources;
    private AtomicBoolean closed = new AtomicBoolean(false);

    private MxNDManager(NDManager parent, Device device) {
        this.parent = parent;
        this.device = Device.defaultIfNull(device);
        resources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
    }

    static MxNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    public MxNDArray create(Pointer handle) {
        MxNDArray array = new MxNDArray(this, handle);
        attach(array.getUid(), array);
        return array;
    }

    public MxSparseNDArray create(Pointer handle, SparseFormat fmt) {
        MxSparseNDArray array = new MxSparseNDArray(this, handle, fmt);
        attach(array.getUid(), array);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDArray create(Shape shape, DataType dataType, Device dev) {
        dev = Device.defaultIfNull(dev, device);
        Pointer handle = JnaUtils.createNdArray(dev, shape, dataType, shape.dimension(), false);
        MxNDArray array = new MxNDArray(this, handle, dev, shape, dataType);
        attach(array.getUid(), array);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public MxSparseNDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device dev) {
        dev = Device.defaultIfNull(dev, device);

        SparseFormat fmt = SparseFormat.CSR;
        DataType dataType = DataType.fromBuffer(data);
        MxNDArray indptrNd = create(new Shape(indptr.length), DataType.INT64, dev);
        indptrNd.set(indptr);
        MxNDArray indicesNd = create(new Shape(indices.length), DataType.INT64, dev);
        indicesNd.set(indices);
        Pointer handle =
                JnaUtils.createSparseNdArray(
                        fmt,
                        dev,
                        shape,
                        dataType,
                        new DataType[] {indptrNd.getDataType(), indicesNd.getDataType()},
                        new Shape[] {indptrNd.getShape(), indicesNd.getShape()},
                        false);
        MxSparseNDArray sparse = create(handle, fmt);
        MxNDArray dataNd = create(new Shape(data.remaining()), dataType, dev);
        dataNd.set(data);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, dataNd, -1);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indptrNd, 0);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indicesNd, 1);
        return sparse;
    }

    /** {@inheritDoc} */
    @Override
    public MxSparseNDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device dev) {
        dev = Device.defaultIfNull(dev, device);

        SparseFormat fmt = SparseFormat.ROW_SPARSE;
        DataType dataType = DataType.fromBuffer(data);
        MxNDArray indicesNd = create(new Shape(indices.length), DataType.INT64, dev);
        indicesNd.set(indices);
        Pointer handle =
                JnaUtils.createSparseNdArray(
                        fmt,
                        dev,
                        shape,
                        dataType,
                        new DataType[] {indicesNd.getDataType()},
                        new Shape[] {indicesNd.getShape()},
                        false);
        MxSparseNDArray sparse = create(handle, fmt);
        MxNDArray dataNd = create(dataShape, dataType, dev);
        dataNd.set(data);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, dataNd, -1);
        JnaUtils.ndArraySyncCopyFromNdArray(sparse, indicesNd, 0);
        return sparse;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device dev) {
        return fill("_npi_zeros", dev, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType, Device dev) {
        return fill("_npi_ones", dev, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(int start, int stop, int step, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("step", step);
        params.setDataType(dataType);
        params.setDevice(Device.defaultIfNull(dev, device));
        return invoke("_npi_arange", EMPTY_LIST, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("N", rows);
        params.addParam("M", cols);
        params.addParam("k", k);
        params.setDataType(dataType);
        params.setDevice(Device.defaultIfNull(dev, device));
        return invoke("_npi_eye", EMPTY_LIST, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(double start, double stop, int num, boolean endpoint, Device dev) {
        if (num < 0) {
            throw new IllegalArgumentException("Num argument must be non-negative");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("num", num);
        params.addParam("endpoint", endpoint);
        params.setDataType(DataType.FLOAT32);
        params.setDevice(Device.defaultIfNull(dev, device));
        return invoke("_npi_linspace", EMPTY_LIST, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            double low, double high, Shape shape, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("low", low);
        params.addParam("high", high);
        params.addParam("size", shape);
        params.setDevice(Device.defaultIfNull(dev, device));
        params.setDataType(dataType);
        return invoke("_npi_uniform", EMPTY_LIST, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            double loc, double scale, Shape shape, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("loc", loc);
        params.addParam("scale", scale);
        params.addParam("size", shape);
        params.setDevice(Device.defaultIfNull(dev, device));
        params.setDataType(dataType);
        return invoke("_npi_normal", EMPTY_LIST, params).singletonOrThrow();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        MxOpParams params = new MxOpParams();
        params.addParam("n", n);
        params.addParam("size", shape);
        return invoke("_npi_multinomial", pValues, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        MxOpParams params = new MxOpParams();
        params.addParam("n", n);
        return invoke("_npi_multinomial", pValues, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public MxNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public MxNDManager newSubManager(Device dev) {
        MxNDManager manager = new MxNDManager(this, dev);
        attach(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void attach(String resourceId, AutoCloseable resource) {
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        WeakReference<AutoCloseable> ref = new WeakReference<>(resource);
        resources.put(resourceId, ref);
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void detach(String resourceId) {
        if (closed.get()) {
            throw new IllegalStateException("NDManager has been closed already.");
        }
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {
        JnaUtils.op(operation)
                .invoke(this, src.toArray(EMPTY_ARRAY), dest.toArray(EMPTY_ARRAY), params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return new NDList(JnaUtils.op(operation).invoke(this, src.toArray(EMPTY_ARRAY), params));
    }

    public NDArray invoke(String operation, NDArray src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(this, src, params)[0];
    }

    public NDArray invoke(String operation, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(this, EMPTY_ARRAY, params)[0];
    }

    @Override
    public String toString() {
        String parentUID = parent == null ? "No Parent" : ((MxNDManager) parent).uid;
        return "UID: "
                + uid
                + " Parent UID: "
                + parentUID
                + " isOpen: "
                + isOpen()
                + " Resource size: "
                + resources.size();
    }

    /** {@inheritDoc} */
    @Override
    public synchronized void close() {
        if (!closed.getAndSet(true)) {
            for (WeakReference<AutoCloseable> resource : resources.values()) {
                AutoCloseable closeable = resource.get();
                if (closeable != null) {
                    try {
                        closeable.close();
                    } catch (Exception ignore) {
                        // ignore
                    }
                }
            }
            parent.detach(uid);
            resources.clear();
        }
    }

    boolean isOpen() {
        return !closed.get();
    }

    private NDArray fill(String opName, Device dev, Shape shape, DataType dataType) {
        MxOpParams params = new MxOpParams();
        if (shape == null) {
            throw new IllegalArgumentException("Shape is required for " + opName.substring(1));
        }
        params.addParam("shape", shape);
        params.setDevice(Device.defaultIfNull(dev, device));
        params.setDataType(dataType);
        return invoke(opName, params);
    }

    private static final class SystemManager extends MxNDManager {

        SystemManager() {
            super(null, Device.defaultDevice());
        }

        /** {@inheritDoc} */
        @Override
        public void attach(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
