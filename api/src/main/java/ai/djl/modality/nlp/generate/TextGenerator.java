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

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.NDScope;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

import java.util.function.Function;
import java.util.stream.Collectors;

public class TextGenerator {

    private String searchName;
    private SearchConfig config;
    private Predictor<NDList, CausalLMOutput> predictor;

    private NDArray positionOffset;

    public TextGenerator(
            Predictor<NDList, CausalLMOutput> predictor,
            String searchName,
            SearchConfig searchConfig) {
        this.predictor = predictor;
        this.searchName = searchName;
        this.config = searchConfig;
    }

    @SuppressWarnings("try")
    public NDArray greedySearch(NDArray inputIds) throws TranslateException {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        GreedyBatchTensorList searchState =
                new GreedyBatchTensorList(inputIds, null, null, attentionMask);
        while (true) {
            try (NDScope ignore = new NDScope()) {
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
                NDList pastKeyValues = searchState.getPastKeyValues();
                if (pastKeyValues != null) {
                    modelInput.addAll(pastKeyValues);
                }
                CausalLMOutput modelOutput = predictor.predict(modelInput);

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

    /**
     * Generates text using beam search.
     *
     * @param inputIds input tokens ids
     * @see https://huggingface.co/blog/how-to-generate
     * @return output tensor
     * @throws TranslateException if failed run forward
     */
    @SuppressWarnings("try")
    public NDArray beamSearch(NDArray inputIds) throws TranslateException {
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        NDManager manager = inputIds.getManager();
        long numBeam = config.getBeam();
        long numBatch = inputIds.getShape().get(0);
        BeamBatchTensorList searchState = new BeamBatchTensorList();

        long numHeads = 0;
        long kvDim = 0;
        while (true) {
            if (searchState.getPastAttentionMask() == null) {
                // Initial beams
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput modelOutput = predictor.predict(modelInput);

                // [batch, probDim]
                NDArray allProbs = modelOutput.getLogits().get(":, -1, :").softmax(1);

                // [batch, beam]
                NDList topK = allProbs.topK(Math.toIntExact(numBeam), -1, true, false);
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
                                modelOutput.getPastKeyValuesList().stream()
                                        .map(fn)
                                        .collect(Collectors.toList()));
                // [batch, beam, seq_past]
                NDArray pastOutputIds = inputIds.expandDims(1).repeat(1, numBeam);

                searchState =
                        new BeamBatchTensorList(
                                outputIds, pastOutputIds, pastKeyValues, attentionMask, lastProbs);

                numHeads = pastKeyValues.get(0).getShape().get(2);
                kvDim = pastKeyValues.get(0).getShape().getLastDimension();
            }

            try (NDScope ignore = new NDScope()) {
                long pastSeqLength = searchState.getPastOutputIds().getShape().getLastDimension();
                NDList modelInput =
                        prepareInput(
                                searchState.getNextInputIds().reshape(numBatch * numBeam, 1),
                                searchState.getPastAttentionMask().reshape(numBatch * numBeam, -1),
                                pastSeqLength,
                                config.getBeam());

                final long finalNumHeads = numHeads;
                final long finalKvDim = kvDim;
                Function<NDArray, NDArray> fn =
                        ndarray ->
                                ndarray.reshape(
                                        numBatch * numBeam,
                                        finalNumHeads,
                                        pastSeqLength,
                                        finalKvDim);
                NDList pastKeyValues =
                        new NDList(
                                searchState.getPastKeyValues().stream()
                                        .map(fn)
                                        .collect(Collectors.toList()));
                modelInput.addAll(pastKeyValues);
                CausalLMOutput modelOutput = predictor.predict(modelInput);

                NDList generatedOutput =
                        StepGeneration.beamStepGeneration(
                                searchState.getLastProbs(),
                                modelOutput.getLogits(),
                                numBatch,
                                numBeam);

                // Update searchState
                searchState = updateSearchState(searchState, modelOutput, generatedOutput, manager);

                // Memory management
                NDScope.unregister(
                        searchState.getNextInputIds(),
                        searchState.getPastOutputIds(),
                        searchState.getPastAttentionMask(),
                        searchState.getLastProbs());
                NDScope.unregister(searchState.getPastKeyValues());
            }

            // Termination Criteria
            // TODO: <EOS>, delete the sentence and add it to result.
            if (searchState.getPastOutputIds().getShape().getLastDimension() + 1
                    >= config.getMaxSeqLength()) {
                break;
            }
        }

        return searchState
                .getPastOutputIds()
                .concat(searchState.getNextInputIds(), -1)
                .reshape(numBatch * numBeam, -1);
    }

    /**
     * Generates text using contrastive search.
     *
     * @param inputIds input token ids
     * @see https://huggingface.co/blog/introducing-csearch
     * @return the generated {@code NDArray}
     * @throws TranslateException if forward failed
     */
    @SuppressWarnings("try")
    public NDArray contrastiveSearch(NDArray inputIds) throws TranslateException {
        // inputIds: [batchSize, seqLength: t_init]
        // attentionMask: [batchSize, pastSeq]. seq-dim-size = |past_seq| + |inputIds|.

        NDManager manager = inputIds.getManager();
        NDArray attentionMask = prepareAttentionMaskOffset(inputIds, config);
        ContrastiveBatchTensorList searchState = new ContrastiveBatchTensorList();
        while (true) {
            if (searchState.getPastKeyValues() == null) {
                NDList modelInput = prepareInput(inputIds, attentionMask, 0, 1);
                CausalLMOutput output = predictor.predict(modelInput);
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

            try (NDScope ignore = new NDScope()) {
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
                candidateModelInput.addAll(kCopyPastKeyValues);
                CausalLMOutput candidateOutput = predictor.predict(candidateModelInput);

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

    private static BeamBatchTensorList updateSearchState(
            BeamBatchTensorList searchState,
            CausalLMOutput modelOutput,
            NDList generatedOutput,
            NDManager manager) {

        NDList pastKeyValues = searchState.getPastKeyValues();
        long numHeads = pastKeyValues.get(0).getShape().get(2);
        long kvDim = pastKeyValues.get(0).getShape().getLastDimension();
        long numBatch = searchState.getPastOutputIds().getShape().get(0);
        long numBeam = searchState.getPastOutputIds().getShape().get(1);
        long pastSeqLength = searchState.getPastOutputIds().getShape().getLastDimension();

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
                searchState
                        .getPastOutputIds()
                        .concat(searchState.getNextInputIds(), -1)
                        .get(sourceBeamIndex);
        Function<NDArray, NDArray> fn =
                ndarray ->
                        ndarray.reshape(numBatch, numBeam, numHeads, pastSeqLength + 1, kvDim)
                                .get(sourceBeamIndex);
        pastKeyValues =
                new NDList(
                        modelOutput.getPastKeyValuesList().stream()
                                .map(fn)
                                .collect(Collectors.toList()));

        NDArray pastAttentionMask =
                searchState
                        .getPastAttentionMask()
                        .concat(manager.ones(new Shape(numBatch, numBeam, 1), DataType.INT64), -1)
                        .get(sourceBeamIndex);

        return new BeamBatchTensorList(
                nextInputIds, pastOutputIds, pastKeyValues, pastAttentionMask, newProbs);
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

    public NDArray forward(NDArray inputIds) throws TranslateException {
        switch (searchName) {
            case "greedy":
                return greedySearch(inputIds);
            case "beam":
                return beamSearch(inputIds);
            case "contrastive":
                return contrastiveSearch(inputIds);
            default:
                throw new IllegalArgumentException(
                        "searchName not correctly specified. Please choose among: {greedy, beam,"
                                + " contrastive}");
        }
    }
}
