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

    public NDArray positionOffset;

    public LMSearch(LMAdapter lmAdapter) {
        this.lmAdapter = lmAdapter;
    }

    public NDArray greedySearch(NDArray inputIds, SearchConfig config) {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        GreedySearchState searchState = new GreedySearchState(inputIds, null, null, attentionMask);
        while (true) {
            long pastSeqLength =
                    searchState.pastOutputIds == null
                            ? 0
                            : searchState.pastOutputIds.getShape().get(-1);
            NDList modelInput =
                    prepareInput(
                            searchState.nextInputIds,
                            searchState.pastAttentionMask,
                            pastSeqLength,
                            1);
            CausalLMOutput modelOutput =
                    lmAdapter.forward(modelInput, searchState.pastKeyValues, manager);

            NDArray outputIds = StepGeneration.GreedyStepGen(modelOutput.logits);

            // Update searchState
            if (searchState.pastOutputIds == null) {
                searchState.pastOutputIds = searchState.nextInputIds;
            } else {
                searchState.pastOutputIds =
                        searchState.pastOutputIds.concat(searchState.nextInputIds, 1);
            }
            searchState.nextInputIds = outputIds;
            searchState.pastKeyValues = modelOutput.pastKeyValuesList;
            searchState.pastAttentionMask =
                    searchState.pastAttentionMask.concat(
                            manager.ones(new Shape(inputIds.getShape().get(0), 1), DataType.INT64),
                            1);

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(1) + 1 >= config.maxSeqLength) {
                break;
            }
        }
        return searchState.pastOutputIds.concat(searchState.nextInputIds, 1);
    }

    public NDArray beamSearch(NDArray inputIds, SearchConfig config) {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        long numBeam = config.beam;
        long numBatch = inputIds.getShape().get(0);
        BeamSearchState searchState = new BeamSearchState();

        long numHeads = 0;
        long kvDim = 0;
        while (true) {
            if (searchState.pastAttentionMask == null) {
                // Initial beams
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput modelOutput = lmAdapter.forward(modelInput, null, manager);

                // [batch, probDim]
                NDArray allProbs = modelOutput.logits.get(":, -1, :").softmax(1);

                // [batch, beam]
                NDList topK = allProbs.topK((int) numBeam, -1, true, false);
                NDArray outputIds = topK.get(1).expandDims(2);
                NDArray lastProbs = topK.get(0).normalize(1, 1);
                assert outputIds.getShape().getShape().length == 3 : "Wrong shape";
                assert lastProbs.getShape().getShape().length == 2 : "Wrong Shape";

                // [batch, beam, seq + 1]
                attentionMask =
                        attentionMask
                                .concat(manager.ones(new Shape(numBatch, 1), DataType.INT64), -1)
                                .expandDims(1)
                                .repeat(1, numBeam);

                // [batch, beam, heads, seq_past, kvFeature]
                Function<NDArray, NDArray> fn = ndarray -> ndarray.expandDims(1).repeat(1, numBeam);
                NDList pastKeyValues =
                        new NDList(
                                modelOutput.pastKeyValuesList.stream()
                                        .map(fn)
                                        .collect(Collectors.toList()));
                // [batch, beam, seq_past]
                NDArray pastOutputIds = inputIds.expandDims(1).repeat(1, numBeam);

                searchState =
                        new BeamSearchState(
                                outputIds, pastOutputIds, pastKeyValues, attentionMask, lastProbs);

                numHeads = pastKeyValues.get(0).getShape().get(-3);
                kvDim = pastKeyValues.get(0).getShape().get(-1);
            }

            long pastSeqLength = searchState.pastOutputIds.getShape().get(-1);
            NDList modelInput =
                    prepareInput(
                            searchState.nextInputIds.reshape(numBatch * numBeam, 1),
                            searchState.pastAttentionMask.reshape(numBatch * numBeam, -1),
                            pastSeqLength,
                            config.beam);

            final long finalNumHeads = numHeads;
            final long finalKvDim = kvDim;
            Function<NDArray, NDArray> fn =
                    ndarray ->
                            ndarray.reshape(
                                    numBatch * numBeam, finalNumHeads, pastSeqLength, finalKvDim);
            NDList pastKeyValues =
                    new NDList(
                            searchState.pastKeyValues.stream()
                                    .map(fn)
                                    .collect(Collectors.toList()));
            CausalLMOutput modelOutput = lmAdapter.forward(modelInput, pastKeyValues, manager);

            NDList generatedOutput =
                    StepGeneration.BeamStepGeneration(
                            searchState, modelOutput.logits, numBatch, numBeam);

            // Update searchState
            searchState = updateSearchState(searchState, modelOutput, generatedOutput, manager);

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.pastOutputIds.getShape().get(-1) + 1 >= config.maxSeqLength) {
                break;
            }
        }

        return searchState
                .pastOutputIds
                .concat(searchState.nextInputIds, -1)
                .reshape(numBatch * numBeam, -1);
    }

    public NDArray contrastiveSearch(
            NDManager manager,
            NDArray inputIds,
            long[][] attentionMaskSlices, // [batch, startIndex, inputIds.getShape.get(1)-1]
            SearchConfig config) {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.

        //        NDList result = new NDList((int) inputIds.getShape().get(0));
        //        NDArray unfinishedBatchIndex =
        // manager.arange(inputIds.getShape().get(0)).reshape(-1, 1);

        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        ContrastiveSearchState searchState = new ContrastiveSearchState();
        while (true) {
            if (searchState.pastKeyValues == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
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
            assert kCopyPastKeyValues.get(0).getShape().get(-2) + 1
                            == kCopyPastAttentionMask.getShape().get(-1)
                    : "attentionMask_seq = past_seq + new_input_seq";

            // Forward with candidates in batch input
            NDList candidateModelInput =
                    prepareInput(
                            candidateInputIds,
                            kCopyPastAttentionMask,
                            searchState.pastOutputIds.getShape().get(-1),
                            config.k);
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

    private BeamSearchState updateSearchState(
            BeamSearchState searchState,
            CausalLMOutput modelOutput,
            NDList generatedOutput,
            NDManager manager) {

        NDList pastKeyValues = searchState.pastKeyValues;
        long numHeads = pastKeyValues.get(0).getShape().get(-3);
        long kvDim = pastKeyValues.get(0).getShape().get(-1);
        long numBatch = searchState.pastOutputIds.getShape().get(0);
        long numBeam = searchState.pastOutputIds.getShape().get(1);
        long pastSeqLength = searchState.pastOutputIds.getShape().get(-1);

        NDArray nextInputIds = generatedOutput.get(0);
        assert nextInputIds.getShape().getShape().length == 3 : "Wrong Shape";
        NDArray newProbs = generatedOutput.get(1);
        // [batch, beamNew]
        NDArray sourceBeamSelected = generatedOutput.get(2);
        // Act on [batch, beam, ...] dimension and the output will be [batch, beam, ...]
        NDIndex sourceBeamIndex =
                new NDIndex(
                        "{}, {}, ...",
                        manager.arange(0, numBatch, 1, DataType.INT64)
                                .expandDims(1)
                                .repeat(1, numBeam),
                        sourceBeamSelected);

        // A simple concatenation is not enough. During the beam selection process, some source
        // beams are selected several times while some source beams are not selected even once.
        // The pastOutput should be reselected to have the right correspondence to the
        // newInputIds.
        NDArray pastOutputIds =
                searchState.pastOutputIds.concat(searchState.nextInputIds, -1).get(sourceBeamIndex);
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, numBeam, numHeads, pastSeqLength + 1, kvDim)
                                .get(sourceBeamIndex);
        pastKeyValues =
                new NDList(
                        modelOutput.pastKeyValuesList.stream()
                                .map(fn)
                                .collect(Collectors.toList()));

        NDArray pastAttentionMask =
                searchState
                        .pastAttentionMask
                        .concat(manager.ones(new Shape(numBatch, numBeam, 1), DataType.INT64), -1)
                        .get(sourceBeamIndex);

        return new BeamSearchState(
                nextInputIds, pastOutputIds, pastKeyValues, pastAttentionMask, newProbs);
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

        // [batch, 1]
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

    private NDArray prepareAttentionMaskOffset(NDArray inputIds, SearchConfig config) {
        // prepare attentionMask and positionOffset
        // Used to initialize the search
        boolean suffixPadding = config.suffixPadding;
        NDManager manager = inputIds.getManager();
        int numBatch = (int) inputIds.getShape().get(0);
        int initSeqSize = (int) inputIds.getShape().get(1);
        NDArray attentionMask =
                manager.ones(new Shape(1, inputIds.getShape().get(-1)), DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        // Linear search from left to find the first position that's not padTokenId.
        long[][] offset = new long[numBatch][1];
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (suffixPadding && aSequence[idx] == config.padTokenId
                        || !suffixPadding && aSequence[idx] != config.padTokenId) {
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
                offset[i][0] = -idx;
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
                                pastSeqLength + inputIds.getShape().get(-1),
                                1,
                                DataType.INT64)
                        .expandDims(0)
                        .repeat(0, inputIds.getShape().get(0));

        NDArray positionIdsShifted = positionIds.addi(positionOffset.repeat(0, repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());

        return new NDList(inputIds, positionIds, attentionMask);
    }
}

class GreedySearchState {
    // [batch, 1]
    public NDArray nextInputIds;

    // [batch, seq_past + new_seq]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask; // can be spared

    /* Variables below are one time step behind the above state variables. Ie, they contain all the past sequence but excludes the time step that corresponds to the above input. */

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    public long pastSeqLength;

    GreedySearchState(
            NDArray nextInputIds,
            NDArray pastOutputIds,
            NDList pastKeyValues,
            NDArray pastAttentionMask) {
        this.nextInputIds = nextInputIds;
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
    }
}

class BeamSearchState {
    // [batch, beam, seq=1]
    public NDArray nextInputIds;

    // [batch, beam]
    public NDArray lastProbs;

    // [batch, beam, seq_past + new_seq]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastAttentionMask; // can be spared

    /* Variables below are one time step behind the above state variables. Ie, they contain all the past sequence but excludes the time step that corresponds to the above input. */

    // [batch, beam, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    // (k, v) * numLayer,
    // kv: [batch, beam, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    BeamSearchState() {}

    BeamSearchState(
            NDArray nextInputIds,
            NDArray pastOutputIds,
            NDList pastKeyValues,
            NDArray pastAttentionMask,
            NDArray lastProb) {
        this.nextInputIds = nextInputIds;
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
        this.lastProbs = lastProb;
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
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDList pastKeyValues;

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    public NDArray pastOutputIds;

    public long pastSeqLength;

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
        NDArray topkScore = topkScorePart2.muli(1 - alpha).subi(topkScorePart1.muli(alpha));

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
        // logits:  [batch, seq, probDim]
        assert logits.getShape().getShape().length == 3 : "unexpected input";
        logits = logits.get(":, -1, :");
        return logits.argMax(-1).expandDims(1); // [batch, vacDim]
    }

    public static NDList BeamStepGeneration(
            BeamSearchState searchState, NDArray logits, long numBatch, long numBeam) {
        // [batch * beamSource, seq, probDim] -> [batch, beamSource, probDim]
        NDArray allProbs = logits.get(":, -1, :").softmax(1).reshape(numBatch, numBeam, -1);

        // Argmax over the probs in the prob dimension.
        // [batch, beamSource, probDim] -> [batch, beamSource, beamChild]
        NDList topK = allProbs.topK((int) numBeam, -1, true, false);
        NDArray outputIs = topK.get(1);
        NDArray stepProbs = topK.get(0);

        // Chain the probability
        // [batch, beamSource] -> [batch, beamSource, 1]
        NDArray lastProbs = searchState.lastProbs.reshape(numBatch, numBeam, 1);
        // [batch, beamSource, beamChild]
        NDArray newProbs = stepProbs.muli(lastProbs);

        // Argmax over the (beamSource * beamChild) dimension
        topK = newProbs.reshape(numBatch, numBeam * numBeam).topK((int) numBeam, -1, true, false);

        // The select indices act on (beamSource, beamChild) dimension. Decides how the new
        // generated tokenIds correspond to the past tokenIds.
        // [batch, beamNew].
        NDArray select = topK.get(1);
        // Act on [batch, beam, ...] dimension and the output will be [batch, beam, ...]
        NDIndex selectIndex =
                new NDIndex(
                        "{}, {}, ...",
                        logits.getManager()
                                .arange(0, numBatch, 1, DataType.INT64)
                                .expandDims(1)
                                .repeat(1, numBeam),
                        select);

        // [batch, beamNew]
        outputIs = outputIs.reshape(numBatch, numBeam * numBeam).get(selectIndex).expandDims(2);
        // [batch, beamNew]
        newProbs = newProbs.reshape(numBatch, numBeam * numBeam).get(selectIndex).normalize(1, 1);

        /* During the beam selection process, some source beams are selected several times while
        some source beams are not selected even once. The pastOutputs should be reselected to
        have the right correspondence to the newInputIds.
        */
        // [batch, beamNew]
        assert select.getDataType() == DataType.INT64 : "Wrong output! Expect integer division";
        assert select.getShape().getShape().length == 2 : "Wrong size. Expect [batch, beamNew]";
        // For each batch, convert the index1 in beamSource*beamChild dimension to its index2 in
        // beamSource dimension: index2 = index1 / numBeam.
        long[] index = select.toLongArray();
        for (int i = 0; i < index.length; i++) {
            index[i] = Math.floorDiv(index[i], numBeam);
        }
        NDArray sourceBeamSelected =
                logits.getManager().create(index, new Shape(numBatch, numBeam));

        return new NDList(outputIs, newProbs, sourceBeamSelected);
    }
    // TODO: implement pytorch floor_divide.
}
