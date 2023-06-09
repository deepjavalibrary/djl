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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.function.Function;
import java.util.stream.Collectors;

public class LMSearch extends AbstractBlock {

    private String searchName;
    private SearchConfig config;
    private LMBlock lmBlock;

    private NDArray positionOffset;

    public LMSearch(LMBlock lmBlock, String searchName, SearchConfig searchConfig) {
        this.lmBlock = lmBlock;
        this.searchName = searchName;
        this.config = searchConfig;
    }

    public NDArray greedySearch(NDArray inputIds) {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        GreedyBatchTensorList searchState =
                new GreedyBatchTensorList(inputIds, null, null, attentionMask);
        while (true) {
            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();

                long pastSeqLength =
                        searchState.getPastOutputIds() == null
                                ? 0
                                : searchState.getPastOutputIds().getShape().getLastDimension();
                NDList modelInput =
                        prepareInput(
                                searchState.getNextInputIds(),
                                searchState.getPastAttentionMask(),
                                pastSeqLength,
                                1);
                CausalLMOutput modelOutput =
                        lmBlock.forward(modelInput, searchState.getPastKeyValues(), manager);

                NDArray outputIds = StepGeneration.greedyStepGen(modelOutput.getLogits());

                // Update searchState
                if (searchState.getPastOutputIds() == null) {
                    searchState.setPastOutputIds(searchState.getNextInputIds());
                } else {
                    searchState.setPastOutputIds(
                            searchState
                                    .getPastOutputIds()
                                    .concat(searchState.getNextInputIds(), 1));
                }
                searchState.setNextInputIds(outputIds);
                searchState.setPastKeyValues(modelOutput.getPastKeyValuesList());
                searchState.setPastAttentionMask(
                        searchState
                                .getPastAttentionMask()
                                .concat(
                                        manager.ones(
                                                new Shape(inputIds.getShape().get(0), 1),
                                                DataType.INT64),
                                        1));

                // memory management
                NDScope.unregister(
                        searchState.getNextInputIds(),
                        searchState.getPastAttentionMask(),
                        searchState.getPastOutputIds());
                NDScope.unregister(searchState.getPastKeyValues());
            }

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.getPastOutputIds().getShape().get(1) + 1 >= config.getMaxSeqLength()) {
                break;
            }
        }
        return searchState.getPastOutputIds().concat(searchState.getNextInputIds(), 1);
    }

    // https://huggingface.co/blog/introducing-csearch
    public NDArray contrastiveSearch(NDArray inputIds) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.

        NDManager manager = inputIds.getManager();
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        ContrastiveBatchTensorList searchState = new ContrastiveBatchTensorList();
        while (true) {
            if (searchState.getPastKeyValues() == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput output = lmBlock.forward(modelInput, null, manager);
                NDArray lastLogits = output.getLogits().get(":, -1, :");
                searchState =
                        new ContrastiveBatchTensorList(
                                inputIds,
                                attentionMask,
                                output.getHiddenState(),
                                lastLogits,
                                output.getPastKeyValuesList(),
                                new long[] {});
            }

            /* Contrastive search loop main part */
            // (1) candidate tokens recall;
            // (2) candidate re-rank by degeneration penalty

            try (NDScope scope = new NDScope()) {
                scope.suppressNotUsedWarning();

                NDArray topKIds =
                        searchState
                                .getLogits()
                                .topK(config.getK(), -1, true, false)
                                .get(1); // [batch, topK]

                // Generate model inputs and put candidates together into batch
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
                                manager.ones(
                                        new Shape(numBatch * config.getK(), 1), DataType.INT64),
                                1);
                assert kCopyPastKeyValues.get(0).getShape().get(2) + 1
                                == kCopyPastAttentionMask.getShape().getLastDimension()
                        : "attentionMask_seq = past_seq + new_input_seq";

                // Forward with candidates in batch input
                NDList candidateModelInput =
                        prepareInput(
                                candidateInputIds,
                                kCopyPastAttentionMask,
                                searchState.getPastOutputIds().getShape().getLastDimension(),
                                config.getK());
                CausalLMOutput candidateOutput =
                        lmBlock.forward(candidateModelInput, kCopyPastKeyValues, manager);

                NDList generatedOutput =
                        StepGeneration.constrastiveStepGeneration(
                                topKIds,
                                searchState.getLogits(),
                                searchState.getPastHiddenStates(),
                                candidateOutput.getHiddenState(),
                                positionOffset,
                                config.getAlpha());

                // Update searchState for next loop
                searchState =
                        updateSearchState(searchState, candidateOutput, generatedOutput, manager);

                // Memory
                NDScope.unregister(
                        searchState.getPastOutputIds(),
                        searchState.getPastAttentionMask(),
                        searchState.getLogits(),
                        searchState.getPastHiddenStates());
                NDScope.unregister(searchState.getPastKeyValues());
            }

            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.getPastOutputIds().getShape().get(1) >= config.getMaxSeqLength()) {
                break;
            }
        }

        return searchState.getPastOutputIds();
    }

    private static ContrastiveBatchTensorList updateSearchState(
            ContrastiveBatchTensorList searchState,
            CausalLMOutput candidateOutput,
            NDList generatedOutput,
            NDManager manager) {
        // Update searchState for next iteration
        assert candidateOutput.getLogits().getShape().get(1) == 1
                : "dimension check: here, outputLogits corresponds to inputSeq == 1";
        long numBatch = searchState.getLogits().getShape().get(0);
        long logitsDim = searchState.getLogits().getShape().get(1);
        long pastSeqLengthPriorUpdate = searchState.getPastOutputIds().getShape().get(1);
        long numHeads = searchState.getPastKeyValues().get(0).getShape().get(1);
        long kvDim = searchState.getPastKeyValues().get(0).getShape().get(3);
        long hiddenDim = searchState.getPastHiddenStates().getShape().get(2);
        long k = candidateOutput.getLogits().getShape().get(0) / numBatch;

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
                candidateOutput.getLogits().reshape(numBatch, k, logitsDim).get(selectIndex);

        // Take from candidateOutput
        // [batch * k, heads, seq_past, feature] --select--> [batch, heads, seq_past, feature]
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, k, numHeads, pastSeqLengthPriorUpdate + 1, kvDim)
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
                                newHiddenState.reshape(numBatch, k, 1, hiddenDim).get(selectIndex),
                                1);

        // To be concatenated into searchState.outputIds
        // [batch, seq_past]
        NDArray outputIds = generatedOutput.get(0);
        NDArray nextOutputIds = searchState.getPastOutputIds().concat(outputIds, 1);

        // [batch, seq_past]
        NDArray nextPastAttentionMask =
                searchState
                        .getPastAttentionMask()
                        .concat(manager.ones(new Shape(numBatch, 1), DataType.INT64), 1);

        return new ContrastiveBatchTensorList(
                nextOutputIds,
                nextPastAttentionMask,
                nextPastHiddenStates,
                nextLogits,
                nextPastKeyValue,
                new long[] {});
    }

    private NDArray prepareAttentionMaskOffset(NDArray inputIds, SearchConfig config) {
        // prepare attentionMask and positionOffset
        // Used to initialize the search
        boolean suffixPadding = config.isSuffixPadding();
        NDManager manager = inputIds.getManager();
        int numBatch = Math.toIntExact(inputIds.getShape().get(0));
        int initSeqSize = Math.toIntExact(inputIds.getShape().get(1));
        NDArray attentionMask =
                manager.ones(new Shape(1, inputIds.getShape().getLastDimension()), DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        // Linear search from left to find the first position that's not padTokenId.
        long[][] offset = new long[numBatch][1];
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (suffixPadding && aSequence[idx] == config.getPadTokenId()
                        || !suffixPadding && aSequence[idx] != config.getPadTokenId()) {
                    break;
                }
                idx++;
            }
            attentionMask.set(
                    new NDIndex(
                            "{},{}:{}",
                            i,
                            suffixPadding ? idx : 0,
                            suffixPadding ? initSeqSize : idx),
                    0);
            if (!suffixPadding) {
                offset[i][0] = idx;
            }
        }
        positionOffset = manager.create(offset);
        return attentionMask;
    }

    private NDList prepareInput(
            NDArray inputIds, NDArray attentionMask, long pastSeqLength, int repeat) {
        // Pack the model input
        NDArray positionIds =
                inputIds.getManager()
                        .arange(
                                pastSeqLength,
                                pastSeqLength + inputIds.getShape().getLastDimension(),
                                1,
                                DataType.INT64)
                        .expandDims(0)
                        .repeat(0, inputIds.getShape().get(0));

        NDArray positionIdsShifted = positionIds.subi(positionOffset.repeat(0, repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());

        return new NDList(inputIds, positionIds, attentionMask);
    }

    public NDArray forward(NDArray inputIds) {
        switch (searchName) {
            case "greedy":
                return greedySearch(inputIds);
            case "contrastive":
                return contrastiveSearch(inputIds);
            default:
                throw new IllegalArgumentException(
                        "searchName not correctly specified. Please choose among: {greedy, beam,"
                                + " contrastive}");
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return new NDList(forward(inputs.get(0)));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {};
    }
}
