package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.tensorflow.Tensors;

public class TfNDFactory implements NDFactory, AutoCloseable {

    public static final TfNDFactory SYSTEM_FACTORY = new SystemFactory();

    private NDFactory parent;
    private Context context;
    private Map<AutoCloseable, AutoCloseable> resources;

    private TfNDFactory(NDFactory parent, Context context) {
        this.parent = parent;
        this.context = context;
        resources = new ConcurrentHashMap<>();
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

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data) {
        return new TfNDArray(Tensors.create(data));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(DataDesc dataDesc) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(
            Context context, Shape shape, DataType dataType, SparseFormat sparseFormat) {
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
    public NDFactory newSubFactory() {
        return newSubFactory(context);
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory newSubFactory(Context context) {
        TfNDFactory factory = new TfNDFactory(this, context);
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
            super(null, Context.defaultContext());
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
