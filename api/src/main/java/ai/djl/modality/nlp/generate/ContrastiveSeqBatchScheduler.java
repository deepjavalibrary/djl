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
package ai.djl.modality.nlp.generate;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.Arrays;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ContrastiveSeqBatchScheduler extends SeqBatchScheduler {

    public ContrastiveSeqBatchScheduler(LMBlock lmBlock, SearchConfig config) {
        super(lmBlock, config);
    }

    @Override
    public SeqBatcher initForward(NDArray inputIds, NDArray batchUids) {
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            manager = inputIds.getManager();
            NDArray initOffSets = computeOffSets(inputIds, config);
            NDArray attentionMask = computeAttentionMask(inputIds, config);
            NDArray positionIds = computePositionIds(inputIds, initOffSets, 0, 1);

            CausalLMOutput output =
                    lmBlock.forward(
                            new NDList(inputIds, positionIds, attentionMask), null, manager);
            NDArray lastLogits = output.getLogits().get(":, -1, :");

            // Used to mark the sequence dimension's ordinal number for each tensor in the
            // serialized
            // batchTensorList
            long[] seqDimOrder = new long[28];
            Arrays.fill(seqDimOrder, 0, 3, 1);
            seqDimOrder[3] = -1; // -1 means no sequence dimension
            Arrays.fill(seqDimOrder, 4, seqDimOrder.length, 2);

            BatchTensorList batchTensorList =
                    new ContrastiveBatchTensorList(
                            inputIds,
                            attentionMask,
                            output.getHiddenState(),
                            lastLogits,
                            output.getPastKeyValuesList(),
                            seqDimOrder);
            SeqBatcher ret = new SeqBatcher(batchTensorList, batchUids, initOffSets, manager);

            // memory management
            NDScope.unregister(output.getPastKeyValuesList());
            NDScope.unregister(output.getHiddenState(), attentionMask, lastLogits);
            NDScope.unregister(ret.offSets, ret.batchUid);

            return ret;
        }
    }

    @Override
    public NDArray inferenceCall() {
        NDArray outputIds;
        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();

            /* Prepare input for one inference call */
            NDArray logits = ((ContrastiveBatchTensorList) seqBatcher.getData()).getLogits();
            NDArray topKIds = logits.topK(config.getK(), -1, true, false).get(1); // [batch, topK]
            ContrastiveBatchTensorList searchState = (ContrastiveBatchTensorList) seqBatcher.data;

            // Embed the topk dimension into batch dimension for an inference all
            // [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
            NDArray candidateInputIds = topKIds.flatten().reshape(-1, 1);
            assert candidateInputIds.getDataType() == DataType.INT64
                    : "inputIds datatype should be int64";
            assert candidateInputIds.getShape().getShape().length == 2 : "shape not right";

            // [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
            NDList kCopyPastKeyValues =
                    new NDList(
                            searchState.getPastKeyValues().stream()
                                    .map(ndarray -> ndarray.repeat(0, config.getK()))
                                    .collect(Collectors.toList()));
            assert kCopyPastKeyValues.get(0).getDataType() == DataType.FLOAT32
                    : "inputIds datatype should be Float32";

            // [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
            long numBatch = topKIds.getShape().get(0);
            NDArray kCopyPastAttentionMask =
                    searchState.getPastAttentionMask().repeat(0, config.getK());
            kCopyPastAttentionMask =
                    kCopyPastAttentionMask.concat(
                            manager.ones(new Shape(numBatch * config.getK(), 1), DataType.INT64),
                            1);
            assert kCopyPastKeyValues.get(0).getShape().get(2) + 1
                            == kCopyPastAttentionMask.getShape().getLastDimension()
                    : "attentionMask_seq = past_seq + new_input_seq";

            // Forward with candidates in batch input
            NDArray candidatePositionIds =
                    computePositionIds(
                            candidateInputIds,
                            seqBatcher.offSets,
                            searchState.getPastOutputIds().getShape().getLastDimension(),
                            config.getK());
            CausalLMOutput candidateOutput =
                    lmBlock.forward(
                            new NDList(
                                    candidateInputIds,
                                    candidatePositionIds,
                                    kCopyPastAttentionMask),
                            kCopyPastKeyValues,
                            manager);

            NDList generatedOutput =
                    StepGeneration.constrastiveStepGeneration(
                            topKIds,
                            logits,
                            searchState.getPastHiddenStates(),
                            candidateOutput.getHiddenState(),
                            seqBatcher.offSets,
                            config.getAlpha());

            /* Update searchState for next loop */
            long logitsDim = logits.getShape().get(1);
            long numHeads = searchState.getPastKeyValues().get(0).getShape().get(1);
            long kvDim = searchState.getPastKeyValues().get(0).getShape().get(3);
            long currentSeqLength = searchState.getPastOutputIds().getShape().get(1);
            long hiddenDim = searchState.getPastHiddenStates().getShape().get(2);

            // [batch, 1]
            NDArray select = generatedOutput.get(1);
            NDIndex selectIndex =
                    new NDIndex(
                            "{}, {}, ...",
                            manager.arange(0, numBatch, 1, DataType.INT64),
                            select.flatten());

            // Take from candidateOutput
            // [batch, k, inputSeq=1, logitsDim] --select--> [batch, logitDim]
            NDArray nextLogits =
                    candidateOutput
                            .getLogits()
                            .reshape(numBatch, config.getK(), logitsDim)
                            .get(selectIndex);

            // Take from candidateOutput
            // [batch * k, heads, seq_past, feature] --select--> [batch, heads, seq_past, feature]
            Function<NDArray, NDArray> fn =
                    ndarray ->
                            ndarray.reshape(
                                            numBatch,
                                            config.getK(),
                                            numHeads,
                                            currentSeqLength + 1,
                                            kvDim)
                                    .get(selectIndex);
            NDList nextPastKeyValue =
                    new NDList(
                            candidateOutput.getPastKeyValuesList().stream()
                                    .map(fn)
                                    .collect(Collectors.toList()));

            // To be concatenated into searchState.pastHiddenStates
            // [batch * k, inputSeq=1, hiddenDim]
            NDArray newHiddenState = candidateOutput.getHiddenState();
            assert newHiddenState.getManager() == manager : "possible leaky memory";
            NDArray nextPastHiddenStates =
                    searchState
                            .getPastHiddenStates()
                            .concat(
                                    newHiddenState
                                            .reshape(numBatch, config.getK(), 1, hiddenDim)
                                            .get(selectIndex),
                                    1);

            // To be concatenated into searchState.outputIds
            // [batch, seq_past]
            outputIds = generatedOutput.get(0);
            NDArray nextOutputIds = searchState.getPastOutputIds().concat(outputIds, 1);

            // [batch, seq_past]
            NDArray nextPastAttentionMask =
                    searchState
                            .getPastAttentionMask()
                            .concat(manager.ones(new Shape(numBatch, 1), DataType.INT64), 1);

            seqBatcher.seqLength++;
            seqBatcher.data =
                    new ContrastiveBatchTensorList(
                            nextOutputIds,
                            nextPastAttentionMask,
                            nextPastHiddenStates,
                            nextLogits,
                            nextPastKeyValue,
                            searchState.getSeqDimOrder());

            /* Exit criteria */
            seqBatcher.exitCriteria(outputIds, config.getMaxSeqLength(), config.getEosTokenId());

            // Memory management
            NDScope.unregister(nextOutputIds);
            NDScope.unregister(nextPastAttentionMask);
            NDScope.unregister(nextPastHiddenStates);
            NDScope.unregister(nextLogits);
            NDScope.unregister(nextPastKeyValue);
            NDScope.unregister(outputIds);
        }
        return outputIds;
    }
}
