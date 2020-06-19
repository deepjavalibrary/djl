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
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.SequenceInfo;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
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

    private OrtSession session;

    /**
     * Constructs a {@code OrtSymbolBlock}.
     *
     * <p>You can create a {@code PtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param session the {@link OrtSession} contains the model information
     */
    public OrtSymbolBlock(OrtSession session) {
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
        NDManager inputManager = inputs.get(0).getManager();
        OrtNDManager manager =
                (inputManager instanceof OrtNDManager)
                        ? (OrtNDManager) inputManager
                        : OrtNDManager.getSystemManager().newSubManager();
        List<String> inputNames = new ArrayList<>(session.getInputNames());
        if (inputs.size() != inputNames.size()) {
            throw new IllegalArgumentException("Input mismatch, looking for: " + inputNames);
        }
        Map<String, OnnxTensor> container = new ConcurrentHashMap<>();
        // feed data in to match names
        for (int i = 0; i < inputNames.size(); ++i) {
            OnnxTensor tensor = getTensorFromNDArray(inputs.get(i), manager);
            container.put(inputNames.get(i), tensor);
        }
        try {
            // forward
            OrtSession.Result results = session.run(container);
            return evaluateOutput(results, inputManager, manager);
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    private OnnxTensor getTensorFromNDArray(NDArray array, NDManager manager) {
        if (!(array instanceof OrtNDArray)) {
            ByteBuffer bb = array.toByteBuffer();
            DataType dp = array.getDataType();
            array = manager.create(dp.asDataType(bb), array.getShape(), dp);
        }
        return ((OrtNDArray) array).getTensor();
    }

    private NDList evaluateOutput(
            OrtSession.Result results, NDManager inputManager, OrtNDManager manager) {
        NDList output = new NDList();
        for (Map.Entry<String, OnnxValue> r : results) {
            OnnxValue value = r.getValue();
            if ((value instanceof OnnxTensor)) {
                output.add(getNDArrayFromTensor((OnnxTensor) value, inputManager, manager));
            } else if (value instanceof OnnxSequence) {
                // TODO: avoid memory copying to heap
                output.add(seq2Nd((OnnxSequence) value, inputManager));
            } else {
                throw new UnsupportedOperationException("Unsupported output type! " + r.getKey());
            }
        }
        // destroy all local OrtNDArray if has 2nd engine
        if (inputManager != manager) {
            manager.close();
        }
        return output;
    }

    private NDArray getNDArrayFromTensor(
            OnnxTensor tensor, NDManager inputManager, OrtNDManager manager) {
        if (inputManager instanceof OrtNDManager) {
            return manager.create(tensor);
        } else {
            // Store NDArray to second engine
            NDArray array = manager.create(tensor);
            ByteBuffer bb = array.toByteBuffer();
            DataType dp = array.getDataType();
            return inputManager.create(dp.asDataType(bb), array.getShape(), dp);
        }
    }

    @SuppressWarnings("unchecked")
    private NDArray seq2Nd(OnnxSequence seq, NDManager manager) {
        try {
            List<Object> values = seq.getValue();
            OnnxJavaType type = seq.getInfo().sequenceType;
            Shape shape = new Shape(values.size());
            DataType dp;
            SequenceInfo info = seq.getInfo();
            if (info.sequenceOfMaps) {
                type = info.mapInfo.valueType;
                List<Object> valuesTmp = new ArrayList<>();
                values.forEach(map -> valuesTmp.addAll(((Map<Object, Object>) map).values()));
                shape = new Shape(values.size(), valuesTmp.size() / values.size());
                values = valuesTmp;
            }
            ByteBuffer buffer = ByteBuffer.allocate(values.size() * type.size);
            switch (type) {
                case FLOAT:
                    values.forEach(ele -> buffer.putFloat((Float) ele));
                    buffer.rewind();
                    return manager.create(buffer.asFloatBuffer(), shape, DataType.FLOAT32);
                case DOUBLE:
                    values.forEach(ele -> buffer.putDouble((Double) ele));
                    buffer.rewind();
                    return manager.create(buffer.asDoubleBuffer(), shape, DataType.FLOAT64);
                case BOOL:
                case INT8:
                    dp = (type == OnnxJavaType.BOOL) ? DataType.BOOLEAN : DataType.INT8;
                    values.forEach(ele -> buffer.put((Byte) ele));
                    buffer.rewind();
                    return manager.create(buffer, shape, dp);
                case INT32:
                    values.forEach(ele -> buffer.putInt((Integer) ele));
                    buffer.rewind();
                    return manager.create(buffer.asIntBuffer(), shape, DataType.INT32);
                case INT64:
                    values.forEach(ele -> buffer.putLong((Long) ele));
                    buffer.rewind();
                    return manager.create(buffer.asLongBuffer(), shape, DataType.INT64);
                default:
                    throw new UnsupportedOperationException("type is not supported: " + type);
            }
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
        if (session != null) {
            try {
                session.close();
                session = null;
            } catch (OrtException e) {
                throw new EngineException(e);
            }
        }
    }
}
