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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.CausalLMOutput;
import ai.djl.translate.GPTConfig;
import ai.djl.translate.StepGenerator;
import ai.djl.util.NativeResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GPT2OrtStepGenerator implements StepGenerator {
    Block[] blocks;
    List<ZooModel<NDList, NDList>> models;

    GPTConfig config;

    public GPT2OrtStepGenerator(String[] modelUrls)
            throws ModelNotFoundException, MalformedModelException, IOException {
        config = new GPTConfig();

        // modelUrl can be replaced to local onnx model file
        blocks = new Block[modelUrls.length];
        models = new ArrayList<>();
        for (int i = 0; i < modelUrls.length; i++) {
            Criteria<NDList, NDList> criteria =
                    Criteria.builder()
                            .setTypes(NDList.class, NDList.class)
                            .optModelUrls(modelUrls[i])
                            .optEngine("OnnxRuntime")
                            .optProgress(new ProgressBar())
                            .build();
            ZooModel<NDList, NDList> model = criteria.loadModel();
            blocks[i] = model.getBlock();
            models.add(model);
        }
    }

    private NDList dummyPastKeyValues(NDArray inputIds, NDManager manager) {
        long numBatch = inputIds.getShape().get(0);
        long hiddenSize = config.hiddenSize;
        long numAttentionHeads = config.numAttentionHeads;
        int numLayers = config.numLayers;

        NDArray keyOrValue = manager.zeros(new Shape(numBatch, numAttentionHeads, 1, hiddenSize));
        NDList output = new NDList();
        output.addAll(Collections.nCopies(2 * numLayers, keyOrValue));
        return output;
    }

    @Override
    public CausalLMOutput stepGeneration2(NDList input, NDList pastKeyValues, NDManager manager) {
        NDArray inputIds = input.get(0);
        inputIds.setName("input_ids");
        NDArray attentionMask = input.get(2);
        attentionMask.setName("attention_mask");

        NDArray useCacheBranch = manager.create(new boolean[] {true}, new Shape(1));
        useCacheBranch.setName("use_cache_branch");
        if (pastKeyValues == null) {
            pastKeyValues = dummyPastKeyValues(inputIds, manager);
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

    @Override
    public CausalLMOutput stepGeneration(
            NDList input, NativeResource<Long> pastKeyValues, NDManager manager) {
        return null;
    }

    public void poc(String inputType)
            throws ModelNotFoundException, MalformedModelException, IOException {}

    @Override
    public void close() {
        models.forEach(ZooModel::close);
    }
}
