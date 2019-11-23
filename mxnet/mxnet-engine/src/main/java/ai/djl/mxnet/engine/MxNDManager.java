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
import ai.djl.engine.EngineException;
import ai.djl.mxnet.jna.JnaUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.util.PairList;
import com.sun.jna.Pointer;
import java.lang.ref.Reference;
import java.lang.ref.WeakReference;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** {@code MxNDManager} is the MXNet implementation of {@link NDManager}. */
public class MxNDManager implements NDManager {

    private static final Logger logger = LoggerFactory.getLogger(MxTrainer.class);

    /**
     * A global {@link NDManager} singleton instance.
     *
     * <p>This NDManager is the root of all the other {@code NDManager}s. NDArrays created by this
     * manager are un-managed, so the user has to close them manually. Those NDArrays will be
     * released on GC, and might be run into an out of native memory issue.
     */
    private static final MxNDManager SYSTEM_MANAGER = new SystemManager();

    private static final NDArray[] EMPTY = new NDArray[0];

    private NDManager parent;
    private String uid;
    private Device device;
    private Map<String, Reference<AutoCloseable>> resources;
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

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /**
     * Creates an MxNDArray with the given Native Memory Pointer and attaches to this manager.
     *
     * @param handle the array's native memory pointer
     * @return the created array
     */
    public MxNDArray create(Pointer handle) {
        MxNDArray array = new MxNDArray(this, handle);
        attach(array.getUid(), array);
        return array;
    }

    /**
     * Creates a sparse MxNDArray with the given Native Memory Pointer and attaches to this manager.
     *
     * @param handle the array's native memory pointer
     * @param fmt the sparse format to use
     * @return the created array
     */
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
    public NDArray arange(Number start, Number stop, Number step, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("step", step);
        if (dataType != null) {
            params.setDataType(dataType);
        }
        params.setDevice(Device.defaultIfNull(dev, device));
        return invoke("_npi_arange", params);
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
        return invoke("_npi_eye", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(Number start, Number stop, int num, boolean endpoint, Device dev) {
        if (num < 0) {
            throw new IllegalArgumentException("Num argument must be non-negative");
        }
        MxOpParams params = new MxOpParams();
        params.addParam("start", start);
        params.addParam("stop", stop);
        params.addParam("num", num);
        params.addParam("endpoint", endpoint);
        params.setDevice(Device.defaultIfNull(dev, device));
        return invoke("_npi_linspace", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            Number low, Number high, Shape shape, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("low", low);
        params.addParam("high", high);
        params.addParam("size", shape);
        params.setDevice(Device.defaultIfNull(dev, device));
        if (dataType != null) {
            params.setDataType(dataType);
        }
        return invoke("_npi_uniform", params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            Number loc, Number scale, Shape shape, DataType dataType, Device dev) {
        MxOpParams params = new MxOpParams();
        params.addParam("loc", loc);
        params.addParam("scale", scale);
        params.addParam("size", shape);
        params.setDevice(Device.defaultIfNull(dev, device));
        if (dataType != null) {
            params.setDataType(dataType);
        }
        return invoke("_npi_normal", params);
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
            // This may happen in the middle of MxNDManager.close()
            return;
        }
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {
        JnaUtils.op(operation).invoke(this, src, dest, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return new NDList(JnaUtils.op(operation).invoke(this, src.toArray(EMPTY), params));
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the {@link NDList} of source {@link NDArray}
     * @param dest the {@link NDList} to save output to
     * @param params the parameters to be passed to the native operator
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {
        invoke(operation, src.toArray(EMPTY), dest.toArray(EMPTY), params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the array of source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray[] src, PairList<String, ?> params) {
        return JnaUtils.op(operation).invoke(this, src, params)[0];
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the source {@link NDArray}
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, NDArray src, PairList<String, ?> params) {
        return invoke(operation, new NDArray[] {src}, params);
    }

    /**
     * An engine specific generic invocation to native operator.
     *
     * <p>You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause portability issues. A native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param params the parameters to be passed to the native operator
     * @return the output array of {@link NDArray}
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    public NDArray invoke(String operation, PairList<String, ?> params) {
        return invoke(operation, EMPTY, params);
    }

    /** {@inheritDoc} */
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
            for (Reference<AutoCloseable> resource : resources.values()) {
                AutoCloseable closeable = resource.get();
                if (closeable != null) {
                    try {
                        closeable.close();
                    } catch (Exception e) {
                        logger.error("Resource close failed.", e);
                    }
                }
            }
            parent.detach(uid);
            resources.clear();
        }
    }

    /**
     * Prints information about this {@link NDManager} and all sub-managers to the console.
     *
     * @param level the level of this {@link NDManager} in the hierarchy
     */
    public void debugDump(int level) {
        StringBuilder sb = new StringBuilder(100);
        for (int i = 0; i < level; ++i) {
            sb.append("    ");
        }
        sb.append("\\--- NDManager(")
                .append(uid.substring(24))
                .append(") resource count: ")
                .append(resources.size());

        System.out.println(sb.toString()); // NOPMD
        for (Reference<AutoCloseable> ref : resources.values()) {
            AutoCloseable c = ref.get();
            if (c instanceof MxNDManager) {
                ((MxNDManager) c).debugDump(level + 1);
            }
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

    /** The SystemManager is the root {@link MxNDManager} of which all others are children. */
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
