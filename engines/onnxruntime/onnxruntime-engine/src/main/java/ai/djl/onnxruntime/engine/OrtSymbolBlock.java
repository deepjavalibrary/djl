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

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code OrtSymbolBlock} is the ONNX Runtime implementation of {@link SymbolBlock}.
 *
 * <p>You can create a {@code OrtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
 * String)}.
 */
public class OrtSymbolBlock extends AbstractSymbolBlock implements AutoCloseable {

    private OrtSession session;
    private OrtNDManager manager;

    /**
     * Constructs a {@code OrtSymbolBlock}.
     *
     * <p>You can create a {@code PtSymbolBlock} using {@link ai.djl.Model#load(java.nio.file.Path,
     * String)}.
     *
     * @param session the {@link OrtSession} contains the model information
     * @param manager the {@link NDManager} to holds the NDArray
     */
    public OrtSymbolBlock(OrtSession session, OrtNDManager manager) {
        this.session = session;
        this.manager = manager;
        manager.attachInternal(UUID.randomUUID().toString(), this);
    }

    /** {@inheritDoc} */
    @Override
    public void removeLastBlock() {
        throw new UnsupportedOperationException("ONNX Runtime not supported");
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        List<String> inputNames = new ArrayList<>(session.getInputNames());
        if (inputs.size() != inputNames.size()) {
            throw new IllegalArgumentException("Input mismatch, looking for: " + inputNames);
        }

        Map<String, OnnxTensor> container = new ConcurrentHashMap<>();
        // forward
        try (OrtNDManager sub = (OrtNDManager) manager.newSubManager()) {
            // If input data has name
            if (inputs.get(0).getName() != null) {
                for (NDArray input : inputs) {
                    String name = input.getName();
                    if (name == null) {
                        throw new IllegalArgumentException(
                                "All or none of input tensors must have a name.");
                    }
                    if (!inputNames.contains(name)) {
                        throw new IllegalArgumentException("Invalid input tensor name: " + name);
                    }
                    OrtNDArray ortNDArray = sub.from(input);
                    container.put(name, ortNDArray.getTensor());
                }
            } else {
                // feed data in to match names
                for (int i = 0; i < inputNames.size(); ++i) {
                    OrtNDArray ortNDArray = sub.from(inputs.get(i));
                    container.put(inputNames.get(i), ortNDArray.getTensor());
                }
            }

            OrtSession.Result results = session.run(container);
            NDList ret = evaluateOutput(results);
            ret.attach(inputs.head().getManager());
            return ret;
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        PairList<String, Shape> result = new PairList<>();
        for (String name : session.getInputNames()) {
            result.add(name, null);
        }
        return result;
    }

    private NDList evaluateOutput(OrtSession.Result results) {
        NDList output = new NDList();
        for (Map.Entry<String, OnnxValue> r : results) {
            OnnxValue value = r.getValue();
            if ((value instanceof OnnxTensor)) {
                output.add(manager.createInternal((OnnxTensor) value));
            } else if (value instanceof OnnxSequence) {
                // TODO: avoid memory copying to heap
                OnnxSequence seq = (OnnxSequence) value;
                if (seq.getInfo().isSequenceOfMaps()) {
                    output.add(seq2Nd(seq));
                } else {
                    output.addAll(seq2NdList(seq));
                }
            } else {
                throw new UnsupportedOperationException("Unsupported output type! " + r.getKey());
            }
        }
        return output;
    }

    @SuppressWarnings("unchecked")
    private NDArray seq2Nd(OnnxSequence seq) {
        try {
            List<OnnxMap> values = (List<OnnxMap>) seq.getValue();
            DataType dp;
            List<Object> finalData = new ArrayList<>();
            OnnxJavaType type = seq.getInfo().mapInfo.valueType;
            for (OnnxMap map : values) {
                finalData.addAll(((Map<Object, Object>) map.getValue()).values());
            }
            Shape shape = new Shape(values.size(), finalData.size() / values.size());
            ByteBuffer buffer = ByteBuffer.allocate(finalData.size() * type.size);
            switch (type) {
                case FLOAT:
                    finalData.forEach(ele -> buffer.putFloat((Float) ele));
                    buffer.rewind();
                    return manager.create(buffer.asFloatBuffer(), shape, DataType.FLOAT32);
                case DOUBLE:
                    finalData.forEach(ele -> buffer.putDouble((Double) ele));
                    buffer.rewind();
                    return manager.create(buffer.asDoubleBuffer(), shape, DataType.FLOAT64);
                case BOOL:
                case INT8:
                    dp = (type == OnnxJavaType.BOOL) ? DataType.BOOLEAN : DataType.INT8;
                    finalData.forEach(ele -> buffer.put((Byte) ele));
                    buffer.rewind();
                    return manager.create(buffer, shape, dp);
                case INT32:
                    finalData.forEach(ele -> buffer.putInt((Integer) ele));
                    buffer.rewind();
                    return manager.create(buffer.asIntBuffer(), shape, DataType.INT32);
                case INT64:
                    finalData.forEach(ele -> buffer.putLong((Long) ele));
                    buffer.rewind();
                    return manager.create(buffer.asLongBuffer(), shape, DataType.INT64);
                default:
                    throw new UnsupportedOperationException("type is not supported: " + type);
            }
        } catch (OrtException e) {
            throw new EngineException(e);
        }
    }

    private NDList seq2NdList(OnnxSequence sequence) {
        try {
            NDList list = new NDList();
            for (OnnxValue value : sequence.getValue()) {
                list.add(manager.createInternal((OnnxTensor) value));
            }
            return list;
        } catch (OrtException e) {
            throw new EngineException(e);
        }
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

    /** {@inheritDoc} */
    @Override
    public ParameterList getDirectParameters() {
        throw new UnsupportedOperationException("Not yet supported");
    }
}
