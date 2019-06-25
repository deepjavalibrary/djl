package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;

import java.nio.Buffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.amazon.ai.util.PairList;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

public class TfNDFactory implements NDFactory, AutoCloseable {

    public static final TfNDFactory SYSTEM_FACTORY = new SystemFactory();
    private static int nameAssignment = 1;

    private NDFactory parent;
    private Context context;
    Graph graph;
    Session session;
    private Map<AutoCloseable, AutoCloseable> resources;

    private TfNDFactory(NDFactory parent, Context context, Graph graph) {
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
    public NDArray create(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(DataDesc dataDesc) {
        return null;
    }

    @Override
    public NDArray create(DataDesc dataDesc, Buffer data) {
        return null;
    }

    public TfNDArray create(Tensor<?> tensor) {
        return new TfNDArray(this, tensor);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray[] invoke(String operation, NDArray[] src, NDArray[] dest, PairList<String, String> params) {
        return new NDArray[0];
    }

    @Override
    public NDArray zeros(Shape shape) {
        return null;
    }

    @Override
    public NDArray zeros(Context context, Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(DataDesc dataDesc) {
        return null;
    }

    @Override
    public NDArray ones(Context context, Shape shape, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(DataDesc dataDesc) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory getParentFactory() {
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
    public NDFactory newSubFactory(Context context) {
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
