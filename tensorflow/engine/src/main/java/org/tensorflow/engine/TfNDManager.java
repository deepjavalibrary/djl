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
package org.tensorflow.engine;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import software.amazon.ai.Device;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class TfNDManager implements NDManager, AutoCloseable {

    static final TfNDManager SYSTEM_MANAGER = new SystemManager();
    private static int nameAssignment = 1;

    private NDManager parent;
    private Device device;
    Graph graph;
    Session session;
    private Map<AutoCloseable, AutoCloseable> resources;

    private TfNDManager(NDManager parent, Device device, Graph graph) {
        this.parent = parent;
        this.device = device;
        this.graph = graph;
        resources = new ConcurrentHashMap<>();
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

    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
    }

    @Override
    public NDArray create(float[] data, Shape shape) {
        return new TfNDArray(this, Tensors.create(data));
    }

    @Override
    public NDArray create(Shape shape, DataType dataType, Device device) {
        return null;
    }

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

    @Override
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        return null;
    }

    @Override
    public NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList load(Path path) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path path, NDList ndList) {}

    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return null;
    }

    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(int start, int stop, int step, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(double start, double stop, int num, boolean endpoint, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            double low, double high, Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            double loc, double scale, Shape shape, DataType dataType, Device device) {
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
        resources.put(manager, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(AutoCloseable resource) {
        resources.put(resource, resource);
    }

    /** {@inheritDoc} */
    @Override
    public void detach(AutoCloseable resource) {
        resources.remove(resource);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (AutoCloseable resource : resources.keySet()) {
            try {
                resource.close();
            } catch (Exception ignore) {
                // ignore
            }
        }
        resources = null;
        parent.detach(this);
    }

    private static final class SystemManager extends TfNDManager {

        SystemManager() {
            super(null, Device.defaultDevice(), new Graph());
            session = new Session(graph);
        }

        /** {@inheritDoc} */
        @Override
        public void attach(AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
