package org.tensorflow.engine;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import software.amazon.ai.Context;
import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDFactory;
import software.amazon.ai.ndarray.types.DataDesc;
import software.amazon.ai.ndarray.types.DataType;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

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

    /** {@inheritDoc} */
    @Override
    public NDArray create(DataDesc dataDesc) {
        return null;
    }

    public NDArray create(int data) {
        return new TfNDArray(this, Tensors.create(data));
    }

    @Override
    public NDArray create(Context context, Shape shape, DataType dataType) {
        return null;
    }

    @Override
    public NDArray create(float[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(int[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(double[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(long[] data, Context context, Shape shape) {
        return null;
    }

    @Override
    public NDArray create(byte[] data, Context context, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(DataDesc dataDesc, Buffer data) {
        return null;
    }

    public TfNDArray create(Tensor<?> tensor) {
        return new TfNDArray(this, tensor);
    }

    public TfNDArray create(Shape shape, ByteBuffer data) {
        return new TfNDArray(this, shape, data);
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDArray[] invoke(String operation, NDArray[] src, PairList<String, ?> params) {
        return new NDArray[0];
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
    public NDArray arange(int start, int stop, int step, Context context, DataType dataType) {
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
            double low, double high, Shape shape, Context context, DataType dataType) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            double loc, double scale, Shape shape, Context context, DataType dataType) {
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
