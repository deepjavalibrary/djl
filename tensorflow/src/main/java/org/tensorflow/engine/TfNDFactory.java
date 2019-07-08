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

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDScopedFactory;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public class TfNDFactory implements NDScopedFactory, AutoCloseable {

    public static final TfNDFactory SYSTEM_FACTORY = new SystemFactory();
    private static int nameAssignment = 1;

    private NDScopedFactory parent;
    private Context context;
    Graph graph;
    Session session;
    private Map<AutoCloseable, AutoCloseable> resources;

    private TfNDFactory(NDScopedFactory parent, Context context, Graph graph) {
        this.parent = parent;
        this.context = context;
        this.graph = graph;
        resources = new ConcurrentHashMap<>();
    }

    Graph getGraph() {
        return graph;
    }

    Session getSession() {
        TfNDFactory f = this;
        while (f.session == null) {
            f = (TfNDFactory) f.getParentFactory();
        }
        return f.session;
    }

    static int nextNameAssignment() {
        return nameAssignment++;
    }

    @Override
    public NDArray create(Shape shape, DataType dataType, Context context) {
        return null;
    }

    public NDArray create(int data) {
        return new TfNDArray(this, Tensors.create(data));
    }

    public TfNDArray create(Tensor<?> tensor) {
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(Shape shape, ByteBuffer data) {
        return new TfNDArray(this, shape, data);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(String operation, NDList src, NDList dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    @Override
    public NDArray zeros(Shape shape, DataType dataType, Context context) {
        return null;
    }

    @Override
    public NDArray ones(Shape shape, DataType dataType, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(int start, int stop, int step, DataType dataType, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(double start, double stop, int num, boolean endpoint, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            double low, double high, Shape shape, DataType dataType, Context context) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            double loc, double scale, Shape shape, DataType dataType, Context context) {
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
    public NDScopedFactory getParentFactory() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return context;
    }

    /** {@inheritDoc} */
    @Override
    public TfNDFactory newSubFactory() {
        return (TfNDFactory) newSubFactory(context);
    }

    /** {@inheritDoc} */
    @Override
    public NDScopedFactory newSubFactory(Context context) {
        TfNDFactory factory = new TfNDFactory(this, context, graph);
        resources.put(factory, factory);
        return factory;
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

    private static final class SystemFactory extends TfNDFactory {

        SystemFactory() {
            super(null, Context.defaultContext(), new Graph());
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
