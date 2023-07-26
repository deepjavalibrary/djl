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
package ai.djl.onnxruntime.zoo.nlp.textgeneration;

import ai.djl.modality.nlp.generate.CausalLMOutput;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

/** The {@link ai.djl.translate.Translator} for PyTorch GPT2 model. */
public class OrtGptTranslator implements NoBatchifyTranslator<NDList, CausalLMOutput> {

    private long kvDim;
    private int numAttentionHeads;
    private int numLayers;

    /**
     * Constructs a new instance of {@code PtGptTranslator}.
     *
     * @param kvDim the kv dimension
     * @param numAttentionHeads the number of attention heads
     * @param numLayers the number of layers
     */
    public OrtGptTranslator(long kvDim, int numAttentionHeads, int numLayers) {
        this.kvDim = kvDim;
        this.numAttentionHeads = numAttentionHeads;
        this.numLayers = numLayers;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, NDList input) throws Exception {
        // input = [inputIds, posIds, attnMask]
        NDManager manager = ctx.getNDManager();
        NDArray inputIds = input.get(0);
        inputIds.setName("input_ids");

        NDArray attentionMask = input.get(2);
        attentionMask.setName("attention_mask");

        NDList inputNew;
        if (input.size() == 3) {
            // pastKeyValue == null
            NDArray useCacheBranch = manager.create(new boolean[] {false}, new Shape(1));
            useCacheBranch.setName("use_cache_branch");
            inputNew = new NDList(inputIds, attentionMask, useCacheBranch);
            initialDummyPastKeyValues(inputIds, manager, inputNew);
        } else {
            NDArray useCacheBranch = manager.create(new boolean[] {true}, new Shape(1));
            useCacheBranch.setName("use_cache_branch");
            inputNew = new NDList(inputIds, attentionMask, useCacheBranch);
            inputNew.addAll(input.subNDList(3));
        }

        int offset = 3;
        for (int i = offset; i < numLayers * 2 + offset; i += 2) {
            int order = (i - offset) / 2;
            inputNew.get(i).setName(String.format("past_key_values.%s.key", order));
            inputNew.get(i + 1).setName(String.format("past_key_values.%s.value", order));
        }

        return inputNew;
    }

    /** {@inheritDoc} */
    @Override
    public CausalLMOutput processOutput(TranslatorContext ctx, NDList output) throws Exception {
        return new CausalLMOutput(output.get(0), output.subNDList(1));
    }

    private void initialDummyPastKeyValues(NDArray inputIds, NDManager manager, NDList list) {
        long numBatch = inputIds.getShape().get(0);
        for (int i = 0; i < numLayers * 2; ++i) {
            NDArray array = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, kvDim));
            list.add(array);
        }
    }
}
