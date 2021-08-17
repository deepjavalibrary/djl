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
package ai.djl.paddlepaddle.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractSymbolBlock;
import ai.djl.nn.SymbolBlock;
import ai.djl.paddlepaddle.jni.JniUtils;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.Arrays;

/** {@code PpSymbolBlock} is the PaddlePaddle implementation of {@link SymbolBlock}. */
public class PpSymbolBlock extends AbstractSymbolBlock {

    private PaddlePredictor predictor;
    private String[] inputNames;

    /**
     * Constructs a new {@code PpSymbolBlock} instance.
     *
     * @param predictor {@link PaddlePredictor} that holds the model information.
     */
    public PpSymbolBlock(PaddlePredictor predictor) {
        this.predictor = predictor;
        inputNames = JniUtils.getInputNames(predictor);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        if (inputNames.length != inputs.size()) {
            throw new IllegalArgumentException(
                    "Input number mismatch, requires: " + Arrays.toString(inputNames));
        }
        NDManager inputManager = inputs.head().getManager();
        try (PpNDManager tempManager = PpNDManager.getSystemManager().newSubManager()) {
            boolean foreignEngine =
                    !PpEngine.ENGINE_NAME.equals(inputManager.getEngine().getEngineName());
            PpNDArray[] result =
                    JniUtils.predictorForward(
                            predictor, getInputs(inputs, foreignEngine, tempManager), inputNames);
            return getOutputs(result, foreignEngine, inputManager);
        }
    }

    private PpNDArray[] getInputs(NDList inputs, boolean foreignEngine, PpNDManager tempManager) {
        PpNDArray[] inputArray = new PpNDArray[inputs.size()];
        for (int i = 0; i < inputArray.length; i++) {
            if (foreignEngine) {
                NDArray array = inputs.get(i);
                inputArray[i] =
                        tempManager.create(
                                array.toByteBuffer(), array.getShape(), array.getDataType());
            } else {
                inputArray[i] = (PpNDArray) inputs.get(i);
            }
        }
        return inputArray;
    }

    private NDList getOutputs(PpNDArray[] outputs, boolean foreignEngine, NDManager inputManager) {
        NDList list = new NDList(outputs.length);
        for (PpNDArray output : outputs) {

            if (foreignEngine) {
                list.add(
                        inputManager.create(
                                output.getDataType().asDataType(output.toByteBuffer()),
                                output.getShape(),
                                output.getDataType()));
            } else {
                list.add(output);
            }
        }
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
