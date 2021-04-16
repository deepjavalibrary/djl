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
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import java.util.List;
import java.util.function.Function;

/** Fully connected Feed-Forward network, only applied to the last dimension of the input. */
public class PointwiseFeedForwardBlock extends SequentialBlock {

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
        // add hidden layers with activation
        for (int hiddenSize : hiddenSizes) {
            add(Linear.builder().optBias(true).setUnits(hiddenSize).build());
            add(new LambdaBlock(activationFunction));
        }
        // add output layer without activation
        add(Linear.builder().optBias(true).setUnits(outputSize).build());
    }
}
