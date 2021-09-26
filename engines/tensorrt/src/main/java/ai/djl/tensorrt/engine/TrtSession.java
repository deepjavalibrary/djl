/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tensorrt.engine;

import ai.djl.engine.EngineException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.tensorrt.jni.JniUtils;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.nio.ByteBuffer;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** {@code TrtSession} represents the TensorRT's execution context. */
public class TrtSession extends AbstractBlock implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(JniUtils.class);

    private long session;

    private NDList inputBindings;
    private NDList outputBindings;
    private Shape[] outputShapes;

    TrtSession(TrtNDManager manager, long modelHandle, long session) {
        this.session = session;
        inputNames = Arrays.asList(JniUtils.getInputNames(modelHandle));
        DataType[] inputTypes = JniUtils.getInputDataTypes(modelHandle);
        inputShapes = new Shape[inputTypes.length];
        inputBindings = new NDList(inputTypes.length);
        for (int i = 0; i < inputTypes.length; ++i) {
            String inputName = inputNames.get(i);
            inputShapes[i] = new Shape(JniUtils.getShape(session, inputName));
            int size = Math.toIntExact(inputShapes[i].size() * inputTypes[i].getNumOfBytes());
            ByteBuffer bb = manager.allocateDirect(size);
            JniUtils.bind(session, inputName, bb);
            NDArray array = manager.create(bb, inputShapes[i], inputTypes[i]);
            array.setName(inputName);
            inputBindings.add(array);
        }

        String[] outputNames = JniUtils.getOutputNames(modelHandle);
        DataType[] outputTypes = JniUtils.getOutputDataTypes(modelHandle);
        outputShapes = new Shape[outputNames.length];
        outputBindings = new NDList(outputShapes.length);
        for (int i = 0; i < outputShapes.length; ++i) {
            outputShapes[i] = new Shape(JniUtils.getShape(session, outputNames[i]));
            int size = Math.toIntExact(outputShapes[i].size() * outputTypes[i].getNumOfBytes());
            ByteBuffer bb = manager.allocateDirect(size);
            JniUtils.bind(session, outputNames[i], bb);
            NDArray array = manager.create(bb, outputShapes[i], outputTypes[i]);
            array.setName(outputNames[i]);
            outputBindings.add(array);
        }

        if (logger.isDebugEnabled()) {
            logger.debug("Model information: ");
            for (int i = 0; i < inputTypes.length; ++i) {
                logger.debug(
                        "input_{}[{}]: {}, {}",
                        i,
                        inputNames.get(i),
                        inputTypes[i],
                        inputShapes[i]);
            }
            for (int i = 0; i < outputTypes.length; ++i) {
                logger.debug(
                        "output_{}[{}]: {}, {}",
                        i,
                        outputNames[i],
                        outputTypes[i],
                        outputShapes[i]);
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        int size = inputs.size();
        if (this.inputBindings.size() != size) {
            throw new EngineException(
                    "Unexpected number of inputs: " + size + ", expected: " + inputBindings.size());
        }
        for (int i = 0; i < size; ++i) {
            NDArray array = inputs.get(i);
            NDArray bound = inputBindings.get(i);
            if (bound != array) {
                if (bound.getDataType() != array.getDataType()) {
                    throw new EngineException(
                            "Unexpected input_"
                                    + i
                                    + '['
                                    + bound.getName()
                                    + "] dataType: "
                                    + array.getDataType()
                                    + ", expected: "
                                    + bound.getDataType());
                } else if (!bound.getShape().equals(array.getShape())) {
                    throw new EngineException(
                            "Unexpected input_"
                                    + i
                                    + '['
                                    + bound.getName()
                                    + "] shape: "
                                    + array.getShape()
                                    + ", expected: "
                                    + bound.getShape());
                }
                bound.set(array.toByteBuffer());
            }
        }
        JniUtils.runTrtModel(session);
        return outputBindings;
    }

    /**
     * Returns the input {@code NDList} that bound to TensorRT engine.
     *
     * @return the input {@code NDList} that bound to TensorRT engine
     */
    public NDList getInputBindings() {
        return inputBindings;
    }

    /**
     * Returns the output {@code NDList} that bound to TensorRT engine.
     *
     * @return the output {@code NDList} that bound to TensorRT engine
     */
    public NDList getOutputBindings() {
        return outputBindings;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return outputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        JniUtils.deleteSession(session);
    }
}
