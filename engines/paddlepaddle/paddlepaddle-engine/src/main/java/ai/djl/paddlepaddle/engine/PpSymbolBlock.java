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
    private PpNDManager manager;
    private String[] inputNames;

    /**
     * Constructs a new {@code PpSymbolBlock} instance.
     *
     * @param predictor {@link PaddlePredictor} that holds the model information.
     * @param manager the {@link NDManager} to holds the NDArray
     */
    public PpSymbolBlock(PaddlePredictor predictor, PpNDManager manager) {
        this.predictor = predictor;
        this.manager = manager;
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
        try (PpNDManager sub = manager.newSubManager()) {
            NDList output =
                    JniUtils.predictorForward(predictor, getInputs(sub, inputs), inputNames);
            NDManager inputManager = inputs.head().getManager();
            NDList ret = new NDList();
            for (NDArray array : output) {
                ret.add(inputManager.from(array));
            }
            return ret;
        }
    }

    private PpNDArray[] getInputs(PpNDManager sub, NDList inputs) {
        PpNDArray[] inputArray = new PpNDArray[inputs.size()];
        for (int i = 0; i < inputArray.length; i++) {
            inputArray[i] = sub.from(inputs.get(i));
        }
        return inputArray;
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
