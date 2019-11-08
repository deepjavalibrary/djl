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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class TfNDManager implements NDManager, AutoCloseable {

    static final TfNDManager SYSTEM_MANAGER = new SystemManager();
    private static int nameAssignment = 1;

    private NDManager parent;
    private String uid;
    private Device device;
    Graph graph;
    Session session;
    private Map<String, AutoCloseable> resources;

    private TfNDManager(NDManager parent, Device device, Graph graph) {
        this.parent = parent;
        this.device = device;
        this.graph = graph;
        resources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
    }

    public static TfNDManager newBaseManager() {
        return SYSTEM_MANAGER.newSubManager();
    }

    public static TfNDManager newBaseManager(Device device) {
        return SYSTEM_MANAGER.newSubManager(device);
    }

    Graph getGraph() {
        return graph;
    }

    Session getSession() {
        TfNDManager f = this;
        while (f.session == null) {
            f = (TfNDManager) f.getParentManager();
        }
        return f.session;
    }

    static int nextNameAssignment() {
        return nameAssignment++;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
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

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(
            Number start, Number stop, Number step, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(Number start, Number stop, int num, boolean endpoint, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            Number low, Number high, Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            Number loc, Number scale, Shape shape, DataType dataType, Device device) {
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
        TfNDManager manager = new TfNDManager(this, device, graph);
        resources.put(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(String resourceId, AutoCloseable resource) {
        resources.put(resourceId, resource);
    }

    /** {@inheritDoc} */
    @Override
    public void detach(String resourceId) {
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (AutoCloseable resource : resources.values()) {
            try {
                resource.close();
            } catch (Exception ignore) {
                // ignore
            }
        }
        resources = null;
        parent.detach(uid);
    }

    private static final class SystemManager extends TfNDManager {

        SystemManager() {
            super(null, Device.defaultDevice(), new Graph());
            session = new Session(graph);
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
