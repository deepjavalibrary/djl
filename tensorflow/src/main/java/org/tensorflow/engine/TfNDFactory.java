package org.tensorflow.engine;

import com.amazon.ai.Context;
import com.amazon.ai.ndarray.NDArray;
import com.amazon.ai.ndarray.NDFactory;
import com.amazon.ai.ndarray.types.DataDesc;
import com.amazon.ai.ndarray.types.DataType;
import com.amazon.ai.ndarray.types.Shape;
import com.amazon.ai.ndarray.types.SparseFormat;

public class TfNDFactory implements NDFactory {

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
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Context getContext() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory newSubFactory() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDFactory newSubFactory(Context context) {
        return null;
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
