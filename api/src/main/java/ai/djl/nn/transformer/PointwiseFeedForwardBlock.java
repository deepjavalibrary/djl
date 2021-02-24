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
package ai.djl.nn.transformer;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

/** Fully connected Feed-Forward network, only applied to the last dimension of the input. */
public class PointwiseFeedForwardBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    /**
     * Creates a pointwise feed-forward block.
     *
     * @param hiddenSizes the sizes of the hidden layers
     * @param outputSize the output size
     * @param activationFunction the activation function to use for the hidden layers (not applied
     *     to output)
     */
    public PointwiseFeedForwardBlock(
            List<Integer> hiddenSizes,
            int outputSize,
            Function<NDList, NDList> activationFunction) {
        super(VERSION);
        // add hidden layers with activation
        int count = 0;
        for (int hiddenSize : hiddenSizes) {
            addChildBlock(
                    "linear_" + count, Linear.builder().optBias(true).setUnits(hiddenSize).build());
            addChildBlock("activation_" + count, new LambdaBlock(activationFunction));
            ++count;
        }
        // add output layer without activation
        addChildBlock("output_layer", Linear.builder().optBias(true).setUnits(outputSize).build());
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        for (Block child : children.values()) {
            inputShapes = child.getOutputShapes(manager, inputShapes);
        }
        return inputShapes;
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        inputNames = Collections.singletonList("input");
        if (inputShapes.length != 1) {
            throw new IllegalArgumentException(
                    "Pointwise feed forward blocks can only have one input.");
        }
        // Now that we know the input shape, we can determine the reshape necessary
        // to shape the input and re-shape the output
        Shape inputShape = inputShapes[0];
        if (inputShape.dimension() < 2) {
            throw new IllegalArgumentException(
                    "Pointwise feed forward blocks need an input of at least dimension 2.");
        }
        Shape lastShape = inputShape;
        for (Block child : children.values()) {
            child.initialize(manager, dataType, lastShape);
            lastShape = getOutputShapes(manager, new Shape[] {lastShape})[0];
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        // go through all layers
        NDList layerResult = inputs;
        for (Pair<String, Block> child : getChildren()) {
            layerResult = child.getValue().forward(ps, layerResult, training);
        }
        return layerResult;
    }
}
