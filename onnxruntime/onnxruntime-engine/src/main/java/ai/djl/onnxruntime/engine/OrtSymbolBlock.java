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

package ai.djl.onnxruntime.engine;

import ai.djl.MalformedModelException;
import ai.djl.engine.EngineException;
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
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code OrtSymbolBlock} is the ONNX Runtime implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code OrtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class OrtSymbolBlock implements SymbolBlock, AutoCloseable {

    private OrtNDManager manager;
    private OrtSession session;

    /**
     * Constructs a {@code OrtSymbolBlock}.
     *
     * <p>You can create a {@code PtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param manager the manager to use for the block
     * @param session the {@link OrtSession} contains the model information
     */
    public OrtSymbolBlock(OrtNDManager manager, OrtSession session) {
        this.manager = manager;
        this.session = session;
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        List<String> inputNames = new ArrayList<>(session.getInputNames());
        if (inputs.size() != inputNames.size()) {
            throw new IllegalArgumentException("Input mismatch, looking for: " + inputNames);
        }
        Map<String, OnnxTensor> container = new ConcurrentHashMap<>();
        // feed data in to match names
        for (int i = 0; i < inputNames.size(); ++i) {
            OnnxTensor tensor = ((OrtNDArray) inputs.get(i)).getTensor();
            container.put(inputNames.get(i), tensor);
        }
        try {
            // forward
            OrtSession.Result results = session.run(container);
            NDList output = new NDList();
            for (Map.Entry<String, OnnxValue> r : results) {
                OnnxValue value = r.getValue();
                if (value instanceof OnnxTensor) {
                    output.add(manager.create((OnnxTensor) value));
                } else {
                    throw new UnsupportedOperationException(
                            "Unsupported output type! " + r.getKey());
                }
            }
            return output;
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void setInitializer(Initializer initializer, String paramName) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public boolean isInitialized() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void clear() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public ParameterList getParameters() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        try {
            session.close();
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }
}
