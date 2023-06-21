/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.engine;

import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

import java.util.stream.Collectors;

public class GPT2PtLMBlock extends LMBlock {
    Block[] blocks;
    GPTConfig config;

    public GPT2PtLMBlock(GPTConfig gptConfig, Block[] blocks) {
        config = gptConfig;
        this.blocks = blocks;
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager) {
        // inputIds, positionIds, attentionMask
        long batchSize = input.get(0).getShape().get(0);
        boolean flagDummyKvCach = pastKeyValues == null;

        if (flagDummyKvCach) {
            pastKeyValues = dummyPastKeyValues(input.get(0), manager, config);
            NDArray attentionMask = input.get(2);
            attentionMask =
                    manager.zeros(new Shape(batchSize, 1), DataType.INT64)
                            .concat(attentionMask, -1);
            input = new NDList(input.get(0), input.get(1), attentionMask);
        }

        String tupleName = "past_key_values(" + config.getNumLayers() + ',' + 2 + ')';
        for (NDArray array : pastKeyValues) {
            array.setName(tupleName);
        }
        input.addAll(pastKeyValues);

        NDList output = blocks[0].forward(null, input, false, null);

        NDArray logitsOutput = output.get(0);
        NDList pastKeyValuesOutput = output.subNDList(1, config.getNumLayers() * 2 + 1);
        NDArray hiddenStatesOutput = manager.zeros(new Shape(1));
        if (output.size() > config.getNumLayers() * 2 + 2) {
            hiddenStatesOutput = output.subNDList(config.getNumLayers() * 2 + 1).get(0);
        }

        if (flagDummyKvCach) {
            NDIndex index2 = new NDIndex(":, :, 1:, ...");
            pastKeyValuesOutput =
                    new NDList(
                            pastKeyValuesOutput.stream()
                                    .map(object -> object.get(index2))
                                    .collect(Collectors.toList()));
        }

        return new CausalLMOutput(logitsOutput, hiddenStatesOutput, pastKeyValuesOutput);
    }
}
