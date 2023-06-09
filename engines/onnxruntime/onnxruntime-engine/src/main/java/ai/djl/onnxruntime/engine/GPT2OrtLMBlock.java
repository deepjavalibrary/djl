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
package ai.djl.onnxruntime.engine;

import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.modality.nlp.generate.GPTConfig;
import ai.djl.modality.nlp.generate.LMBlock;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;

public class GPT2OrtLMBlock extends LMBlock {
    Block[] blocks;
    GPTConfig config;

    public GPT2OrtLMBlock(GPTConfig gptConfig, Block[] blocks) {
        config = gptConfig;
        this.blocks = blocks;
    }

    @Override
    public CausalLMOutput forward(NDList input, NDList pastKeyValues, NDManager manager) {
        // inputIds, positionIds, attentionMask
        NDArray inputIds = input.get(0);
        inputIds.setName("input_ids");
        NDArray attentionMask = input.get(2);
        attentionMask.setName("attention_mask");

        NDArray useCacheBranch = manager.create(new boolean[] {true}, new Shape(1));
        useCacheBranch.setName("use_cache_branch");
        if (pastKeyValues == null) {
            pastKeyValues = dummyPastKeyValues(inputIds, manager, config);
            useCacheBranch.set(new NDIndex(0), manager.create(new boolean[] {false}, new Shape(1)));
        }

        int numLayer = pastKeyValues.size() / 2;
        for (int i = 0; i < numLayer; i++) {
            int pairIdx = i * 2;
            pastKeyValues.get(pairIdx).setName(String.format("past_key_values.%s.key", i));
            pastKeyValues.get(pairIdx + 1).setName(String.format("past_key_values.%s.value", i));
        }

        input = new NDList(inputIds, attentionMask);
        input.add(useCacheBranch);
        input.addAll(pastKeyValues);
        NDList output = blocks[0].forward(null, input, false, null);

        return new CausalLMOutput(output.get(0), output.subNDList(1));
    }
}
