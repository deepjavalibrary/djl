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
package org.apache.mxnet.nn;

import software.amazon.ai.Block;
import software.amazon.ai.ndarray.NDList;
import software.amazon.ai.ndarray.NDManager;
import software.amazon.ai.ndarray.types.Shape;
import software.amazon.ai.util.PairList;

public abstract class MxNNBlock implements Block {

    protected String opName;
    private boolean initialized;
    protected Shape inputShape;
    protected Shape inChannels;

    /** {@inheritDoc} */
    @Override
    public NDList forward(NDList inputs, PairList<String, Object> params) {
        ensureInitialized(inputs);
        NDManager manager = inputs.get(0).getManager();
        return manager.invoke(opName, opInputs(inputs), opParams(params).unique());
    }

    protected abstract NDList opInputs(NDList inputs);

    protected abstract PairList<String, Object> opParams(PairList<String, Object> params);

    /** {@inheritDoc} */
    @Override
    public void ensureInitialized(NDList inputs) {
        Block.super.ensureInitialized(inputs);
        initialized = true;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        return initialized;
    }

    /** {@inheritDoc} */
    @Override
    public Shape getInputShape() {
        return inputShape;
    }

    /** {@inheritDoc} */
    @Override
    public void backward() {}

    /** {@inheritDoc} */
    @Override
    public byte[] getEncoded() {
        return new byte[0];
    }
}
