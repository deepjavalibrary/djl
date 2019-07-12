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
package org.apache.mxnet.engine;

import software.amazon.ai.ndarray.NDArray;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.internal.NDArrayEx;

class MxNDArrayEx implements NDArrayEx {

    private MxNDArray array;
    private MxNDManager manager;

    MxNDArrayEx(MxNDArray parent) {
        this.array = parent;
        this.manager = (MxNDManager) parent.getManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_rdiv_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdiv(NDArray b) {
        return b.div(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rdivi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_rdiv_scalar", new NDList(array), new NDList(array), params);
        return array;
    }

    @Override
    public NDArray rdivi(NDArray b) {
        manager.invoke("elemwise_div", new NDList(b, array), new NDList(array), null);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsub(Number n) {
        return array.sub(n).negi();
    }

    @Override
    public NDArray rsub(NDArray b) {
        return array.sub(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rsubi(Number n) {
        return array.subi(n).negi();
    }

    @Override
    public NDArray rsubi(NDArray b) {
        return array.subi(b).negi();
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        return manager.invoke("_npi_rmod_scalar", array, params);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmod(NDArray b) {
        return b.mod(array);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(Number n) {
        MxOpParams params = new MxOpParams();
        params.add("scalar", n.toString());
        manager.invoke("_npi_rmod_scalar", new NDList(array), new NDList(array), params);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray rmodi(NDArray b) {
        manager.invoke("_npi_mod", new NDList(b, array), new NDList(array), null);
        return array;
    }

    // Sgd update function for non-multi-precision
    @Override
    public void sgdUpdate(
            NDArray grad,
            float lr,
            float wd,
            float rescaleGrad,
            float clipGradient,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", lr);
        params.addParam("wd", wd);
        params.addParam("lazy_update", lazyUpdate);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGradient);
        manager.invoke("sgd_update", new NDList(array, grad), new NDList(array), params);
    }

    @Override
    public void sgdMomUpdate(
            NDArray grad,
            NDArray state,
            float lr,
            float wd,
            float momentum,
            float rescaleGrad,
            float clipGradient,
            boolean lazyUpdate) {
        MxOpParams params = new MxOpParams();
        params.addParam("lr", lr);
        params.addParam("wd", wd);
        params.addParam("momentum", momentum);
        params.addParam("lazy_update", lazyUpdate);
        params.addParam("rescale_grad", rescaleGrad);
        params.addParam("clip_gradient", clipGradient);
        manager.invoke("sgd_update", new NDList(array, grad, state), new NDList(array), params);
    }
}
