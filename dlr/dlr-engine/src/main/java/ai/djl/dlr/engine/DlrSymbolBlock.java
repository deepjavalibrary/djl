/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.dlr.engine;

import ai.djl.dlr.jni.JniUtils;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.BlockList;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.NativeResource;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * {@code DlrSymbolBlock} is the DLR implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code DlrSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class DlrSymbolBlock extends NativeResource<Long> implements SymbolBlock {

    /**
     * Constructs a {@code DlrSymbolBlock}.
     *
     * <p>You can create a {@code DlrSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param handle the handle for native DLR model
     */
    public DlrSymbolBlock(long handle) {
        super(handle);
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        long modelHandle = getHandle();
        NDManager manager = inputs.head().getManager();
        // TODO maybe verify the number of inputs
        // currently we assume the order of the input NDList is the same
        // as the model input
        for (int i = 0; i < inputs.size(); ++i) {
            JniUtils.setDlrInput(modelHandle, inputs.get(i), i);
        }
        JniUtils.runDlrModel(modelHandle);
        return JniUtils.getDlrOutputs(modelHandle, manager);
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) {
        throw new UnsupportedOperationException("not supported for DlrSymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        Long pointer = handle.getAndSet(null);
        if (pointer != null) {
            JniUtils.deleteDlrModel(pointer);
        }
    }
}
