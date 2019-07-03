package org.apache.mxnet.engine;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.internal.NDArrayEx;

class MxNDArrayEx implements NDArrayEx {
    private MxNDArray mxNDArray;
    private MxNDFactory factory;

    MxNDArrayEx(MxNDArray parent) {
        this.mxNDArray = parent;
        this.factory = (MxNDFactory) parent.getFactory();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_rdiv_scalar", mxNDArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return b.div(mxNDArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke(
                "_rdiv_scalar", new NDArray[] {mxNDArray}, new NDArray[] {mxNDArray}, params);
        return mxNDArray;
    }

    @Override
    public NDArray rdivi(NDArray b) {
        factory.invoke(
                "elemwise_div", new NDArray[] {b, mxNDArray}, new NDArray[] {mxNDArray}, null);
        return mxNDArray;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return mxNDArray.sub(n).negi();
    }

    @Override
    public NDArray rsub(NDArray b) {
        return mxNDArray.sub(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return mxNDArray.subi(n).negi();
    }

    @Override
    public NDArray rsubi(NDArray b) {
        return mxNDArray.subi(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return factory.invoke("_npi_rmod_scalar", mxNDArray, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return b.mod(mxNDArray);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        factory.invoke(
                "_npi_rmod_scalar", new NDArray[] {mxNDArray}, new NDArray[] {mxNDArray}, params);
        return mxNDArray;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        factory.invoke("_npi_mod", new NDArray[] {b, mxNDArray}, new NDArray[] {mxNDArray}, null);
        return mxNDArray;
    }
}
