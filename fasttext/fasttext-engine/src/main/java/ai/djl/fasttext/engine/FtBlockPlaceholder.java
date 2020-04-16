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
package ai.djl.fasttext.engine;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

/** A placeholder {@link Block} to comply with the API. */
public final class FtBlockPlaceholder implements Block {

    public static final FtBlockPlaceholder PLACEHOLDER = new FtBlockPlaceholder();

    private FtBlockPlaceholder() {}

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return inputs;
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) {
        throw new UnsupportedOperationException("Not support by FtEngine");
    }
}
