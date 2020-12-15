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

package ai.djl.tflite.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.BlockList;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.Initializer;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.Buffer;
import org.tensorflow.lite.Interpreter;

/**
 * {@code TfLiteSymbolBlock} is the TFLite implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code TfLiteSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class TfLiteSymbolBlock implements SymbolBlock, AutoCloseable {

    private TfLiteNDManager manager;
    private Interpreter interpreter;

    public TfLiteSymbolBlock(Interpreter interpreter, TfLiteNDManager manager) {
        this.interpreter = interpreter;
        this.manager = manager;
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        Object[] intInput = inputs.stream().map(NDArray::toByteBuffer).toArray();
        interpreter.runForMultipleInputsOutputs(intInput);

        int outputSize = interpreter.getOutputTensorCount();
        NDList result = new NDList(outputSize);
        for (int i = 0; i < outputSize; i++) {
            result.add(new TfLiteNDArray(manager, interpreter.getOutputTensor(i)));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is) {
        throw new UnsupportedOperationException("TFLite not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        interpreter.close();
    }
}
