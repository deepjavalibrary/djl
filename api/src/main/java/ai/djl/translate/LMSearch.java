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
package ai.djl.translate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.function.Function;
import java.util.stream.Collectors;

public class LMSearch {

    private LMAdapter lmAdapter;

    public LMSearch(LMAdapter lmAdapter) {
        this.lmAdapter = lmAdapter;
    }

    public NDArray contrastiveSearch(
            NDManager manager,
            NDArray inputIds,
            NDArray attentionMask,
            long[][] attentionMaskSlices, // [batch, startIndex, inputIds.getShape.get(1)-1]
            SearchConfig config) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.

        //        NDList result = new NDList((int) inputIds.getShape().get(0));
        //        NDArray unfinishedBatchIndex =
        // manager.arange(inputIds.getShape().get(0)).reshape(-1, 1);

        ContrastiveSearchState searchState = new ContrastiveSearchState();
        while (true) {
            if (searchState.pastKeyValues == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, null, manager);
                CausalLMOutput output = lmAdapter.forward(modelInput, null, manager);
                NDArray lastLogits = output.logits.get(":, -1, :");
                searchState =
                        new ContrastiveSearchState(
                                lastLogits,
                                output.pastKeyValuesList,
                                output.allHiddenStates.get(0),
                                inputIds,
                                attentionMask);
            }

            /* Contrastive search loop main part */
            // (1) candidate tokens recall;
            // (2) candidate re-rank by degeneration penalty

            NDArray topKIds =
                    searchState.logits.topK(config.k, -1, true, false).get(1); // [batch, topK]

            // Generate model inputs and put candidates together into batch
            // [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
            NDArray candidateInputIds = topKIds.flatten().reshape(-1, 1);
            assert candidateInputIds.getDataType() == DataType.INT64
                    : "inputIds datatype should be int64";
            assert candidateInputIds.getShape().getShape().length == 2 : "shape not right";

            // [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
            NDList kCopyPastKeyValues =
                    new NDList(
                            searchState.pastKeyValues.stream()
                                    .map(ndarray -> ndarray.repeat(0, config.k))
                                    .collect(Collectors.toList()));
            assert kCopyPastKeyValues.get(0).getDataType() == DataType.FLOAT32
                    : "inputIds datatype should be Float32";

            // [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
            long numBatch = topKIds.getShape().get(0);
            NDArray kCopyPastAttentionMask = searchState.pastAttentionMask.repeat(0, config.k);
            kCopyPastAttentionMask =
                    kCopyPastAttentionMask.concat(
                            manager.ones(new Shape(numBatch * config.k, 1), DataType.INT64), 1);
            assert kCopyPastKeyValues.get(0).getShape().size(-2)
                            == kCopyPastAttentionMask.getShape().size(-1) + 1
                    : "attentionMask_seq = past_seq + new_input_seq";

            // Forward with candidates in batch input
            NDList candidateModelInput =
                    prepareInput(
                            candidateInputIds, kCopyPastAttentionMask, kCopyPastKeyValues, manager);
            CausalLMOutput candidateOutput =
                    lmAdapter.forward(candidateModelInput, kCopyPastKeyValues, manager);

            NDList generatedOutput =
                    StepGeneration.ConstrastStepGeneration(
                            topKIds,
                            searchState.logits,
                            searchState.pastHiddenStates,
                            candidateOutput.allHiddenStates.get(0),
                            attentionMaskSlices,
                            config.alpha);

            // Update searchState for next loop
            searchState = updateSearchState(searchState, candidateOutput, generatedOutput, manager);

            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(1) >= config.maxSeqLength) {
                break;
            }
        }

        return searchState.pastOutputIds;
    }

    private ContrastiveSearchState updateSearchState(
            ContrastiveSearchState searchState,
            CausalLMOutput candidateOutput,
            NDList generatedOutput,
            NDManager manager) {
        // Update searchState for next iteration
        assert candidateOutput.logits.getShape().get(1) == 1
                : "dimension check: here, outputLogits corresponds to inputSeq == 1";
        long numBatch = searchState.logits.getShape().get(0);
        long logitsDim = searchState.logits.getShape().get(1);
        long pastSeqLengthPriorUpdate = searchState.pastOutputIds.getShape().get(1);
        long numHeads = searchState.pastKeyValues.get(0).getShape().get(1);
        long kvDim = searchState.pastKeyValues.get(0).getShape().get(3);
        long hiddenDim = searchState.pastHiddenStates.getShape().get(2);
        long k = candidateOutput.logits.getShape().get(0) / numBatch;

        NDArray select = generatedOutput.get(1);
        NDIndex selectIndex =
                new NDIndex(
                        "{}, {}, ...",
                        manager.arange(0, numBatch, 1, DataType.INT64),
                        select.flatten());

        // Take from candidateOutput
        // [batch, k, inputSeq=1, logitsDim] -select-> [batch, logitDim]
        NDArray nextLogits =
                candidateOutput.logits.reshape(numBatch, k, logitsDim).get(selectIndex);

        // Take from candidateOutput
        // [batch * k, heads, seq_past, feature] -select-> [batch, heads, seq_past, feature]
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, k, numHeads, pastSeqLengthPriorUpdate + 1, kvDim)
                                .get(selectIndex);
        NDList nextPastKeyValue =
                new NDList(
                        candidateOutput.pastKeyValuesList.stream()
                                .map(fn)
                                .collect(Collectors.toList()));

        // To be concatenated into searchState.pastHiddenStates
        // [batch * k, inputSeq=1, hiddenDim]
        NDArray newHiddenState = candidateOutput.allHiddenStates.get(0);
        assert newHiddenState.getManager() == manager : "possible leaky memory";
        NDArray nextPastHiddenStates =
                searchState.pastHiddenStates.concat(
                        newHiddenState.reshape(numBatch, k, 1, hiddenDim).get(selectIndex), 1);

        // To be concatenated into searchState.outputIds
        // [batch, seq_past]
        NDArray outputIds = generatedOutput.get(0);
        NDArray nextOutputIds = searchState.pastOutputIds.concat(outputIds, 1);

        // [batch, seq_past]
        NDArray nextPastAttentionMask =
                searchState.pastAttentionMask.concat(
                        manager.ones(new Shape(numBatch, 1), DataType.INT64), 1);

        return new ContrastiveSearchState(
                nextLogits,
                nextPastKeyValue,
                nextPastHiddenStates,
                nextOutputIds,
                nextPastAttentionMask); // can be spared.
    }

    private NDList prepareInput(
            NDArray inputIds, NDArray attentionMask, NDList pastKeyValues, NDManager manager) {
        long pastSeqLen = pastKeyValues == null ? 0 : pastKeyValues.get(0).getShape().size(-2);
        NDArray positionIds =
                manager.arange(
                                pastSeqLen,
                                pastSeqLen + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, inputIds.getShape().get(0));

        return new NDList(inputIds, positionIds, attentionMask);
    }
}

class ContrastiveSearchState {
    // [batch, cls]. Only the last logits, used to recall candidate token
    public NDArray logits;

    // [batch, seq_past, hiddenDim]
    // The embed vector of the past seq. seq-dim-size = |past_seq|. Will grow.
    public NDArray pastHiddenStates;

    // [batch, seq_past]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask; // can be spared

    // (k, v) * numLayer,
    // k or v: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    ContrastiveSearchState() {}

    ContrastiveSearchState(
            NDArray logits,
            NDList pastKeyValues,
            NDArray pastHiddenStates,
            NDArray pastOutputIds,
            NDArray pastAttentionMask) {
        this.logits = logits;
        this.pastKeyValues = pastKeyValues;
        this.pastHiddenStates = pastHiddenStates;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
    }
}

final class StepGeneration {
    private StepGeneration() {}

    public static NDList ConstrastStepGeneration(
            NDArray topKIds,
            NDArray logits,
            NDArray contextHiddenStates,
            NDArray topkHiddenStates,
            long[][] attentionMaskSlices,
            float alpha) {
        /*
          topKIds: [batch, topK]
          attentionMask: [batch, past_seq]
          logits:  [batch, cls]
          contextHiddenStates: [batch, past_seq, dim]
          topkHiddenStates: [batch*topK, seq=1, dim]
          attentionMaskSlice: [batch, startIndex, initSeqLength]
        */

        long batch = topKIds.getShape().get(0);
        long topK = topKIds.getShape().get(1);
        long hiddenDim = topkHiddenStates.getShape().get(-1);

        // [batch*topK, seq=1, dim] -> [batch, topK, dim]
        topkHiddenStates = topkHiddenStates.reshape(batch, topK, hiddenDim);

        //  [batch, topK, dim] * [batch, past_seq, dim] -> [batch, topK, past_seq]
        topkHiddenStates = topkHiddenStates.normalize(2, 2);
        contextHiddenStates = contextHiddenStates.normalize(2, 2);
        NDArray cosSimilarity =
                topkHiddenStates.batchMatMul(contextHiddenStates.transpose(0, 2, 1));

        // Deactivate entries (batch_idx, :, zero_attention_idx_slice) in max{cosSim} step
        for (int i = 0; i < attentionMaskSlices.length; i++) {
            cosSimilarity.set(
                    new NDIndex(
                            "{}, :, {}:{}",
                            i,
                            attentionMaskSlices[i][0],
                            attentionMaskSlices[i][1]),
                    -1);
        }

        // [batch, topK, past_seq] -> [batch, topK]
        NDArray topkScorePart1 = cosSimilarity.max(new int[] {2});
        assert topkScorePart1.getShape().getShape().length == 2 : "Wrong output size";
        // [batch, logitDim].gather([batch, topK) -> [batch, topK]
        NDArray topkScorePart2 = logits.softmax(1).gather(topKIds, 1);
        NDArray topkScore = topkScorePart2.mul(1 - alpha).sub(topkScorePart1.mul(alpha));

        // [batch, topK] => [batch, 1]
        NDArray select = topkScore.argMax(1);
        NDIndex selectIndex =
                new NDIndex(
                        "{}, {}, ...",
                        logits.getManager().arange(0, topKIds.getShape().get(0), 1, DataType.INT64),
                        select);
        NDArray outputIds = topKIds.get(selectIndex).reshape(-1, 1);
        return new NDList(outputIds, select);
    }

    // TODO: add support of Einstein summation:
    // a = torch.randn(batch, past_seq, dim)
    // b = torch.randn(batch, topK, dim)
    // result = torch.einsum('bik,bjk->bij', a, b)

    public static NDArray GreedyStepGen(NDArray logits) {
        return null;
    }
}
