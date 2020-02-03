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
package ai.djl.pytorch.engine;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.pytorch.jni.NativeResource;
import ai.djl.pytorch.jni.Pointer;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

// TODO: Memory handling
public class PtSymbolBlock extends NativeResource implements SymbolBlock {

    private PtNDManager manager;

    public PtSymbolBlock(PtNDManager manager, Pointer handle) {
        super(handle);
        this.manager = manager;
        // Set for inference mode by default
        JniUtils.moduleEval(handle);
    }

    @Override
    public void close() {
        // TODO: Implement close methods
    }

    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return JniUtils.moduleForward(getHandle(), inputs);
    }

    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        return new Shape[0];
    }

    @Override
    public boolean isInitialized() {
        return false;
    }

    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public PairList<String, Shape> describeInput() {
        return null;
    }

    @Override
    public BlockList getChildren() {
        return null;
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return null;
    }

    @Override
    public ParameterList getParameters() {
        return null;
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        return null;
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return new Shape[0];
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        throw new UnsupportedOperationException("Not supported for PyTorch");
    }
}
