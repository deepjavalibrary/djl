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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * {@code StepGeneration} is a utility class containing the step generation utility functions used
 * in autoregressive search.
 */
public final class StepGeneration {

    private StepGeneration() {}

    /**
     * Generate the output token id and selecting indices used in contrastive search.
     *
     * @param topKIds the topk candidate token ids
     * @param logits the logits from the language model
     * @param contextHiddenStates the embedding of the past generated token ids
     * @param topkHiddenStates the embedding of the topk candidate token ids
     * @param offSets the offsets
     * @param alpha the repetition penalty
     * @return the output token ids and selecting indices
     */
    public static NDList constrastiveStepGeneration(
            NDArray topKIds,
            NDArray logits,
            NDArray contextHiddenStates,
            NDArray topkHiddenStates,
            NDArray offSets,
            float alpha) {
        /*
          topKIds: [batch, topK]
          attentionMask: [batch, past_seq]
          logits:  [batch, vocabSize]
          contextHiddenStates: [batch, past_seq, dim]
          topkHiddenStates: [batch*topK, seq=1, dim]
          attentionMaskSlice: [batch, 2]: (startPosition, endPosition)
        */

        long batch = topKIds.getShape().get(0);
        long topK = topKIds.getShape().get(1);
        long hiddenDim = topkHiddenStates.getShape().getLastDimension();

        // [batch*topK, seq=1, dim] -> [batch, topK, dim]
        topkHiddenStates = topkHiddenStates.reshape(batch, topK, hiddenDim);

        //  [batch, topK, dim] * [batch, past_seq, dim] -> [batch, topK, past_seq]
        topkHiddenStates = topkHiddenStates.normalize(2, 2);
        contextHiddenStates = contextHiddenStates.normalize(2, 2);
        NDArray cosSimilarity =
                topkHiddenStates.batchMatMul(contextHiddenStates.transpose(0, 2, 1));

        // Deactivate entries (batch_idx, :, zero_attention_idx_slice) in max{cosSim} step
        long[] offSetsArray = offSets.toLongArray();
        for (int i = 0; i < offSetsArray.length; i++) {
            cosSimilarity.set(new NDIndex("{}, :, {}:{}", i, 0, offSetsArray[i]), -1);
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

    /**
     * Generate the output token id for greedy search.
     *
     * @param logits the logits from the language model
     * @return the output token ids
     */
    public static NDArray greedyStepGen(NDArray logits) {
        // logits:  [batch, seq, probDim]
        assert logits.getShape().getShape().length == 3 : "unexpected input";
        logits = logits.get(":, -1, :");
        return logits.argMax(-1).expandDims(1); // [batch, vacDim]
    }

    /**
     * Generate the output token id and selecting indices used in beam search.
     *
     * @param lastProbs the probabilities of the past prefix sequences
     * @param logits the logits
     * @param numBatch number of batch
     * @param numBeam number of beam
     * @return the output token ids and selecting indices
     */
    public static NDList beamStepGeneration(
            NDArray lastProbs, NDArray logits, long numBatch, long numBeam) {
        // [batch * beamSource, seq, probDim] -> [batch, beamSource, probDim]
        NDArray allProbs = logits.get(":, -1, :").softmax(1).reshape(numBatch, numBeam, -1);

        // Argmax over the probs in the prob dimension.
        // [batch, beamSource, probDim] -> [batch, beamSource, beamChild]
        NDList topK = allProbs.topK(Math.toIntExact(numBeam), -1, true, false);
        NDArray outputIs = topK.get(1);
        NDArray stepProbs = topK.get(0);

        // Chain the probability
        // [batch, beamSource] -> [batch, beamSource, 1]
        lastProbs = lastProbs.reshape(numBatch, numBeam, 1);
        // [batch, beamSource, beamChild]
        NDArray newProbs = stepProbs.muli(lastProbs);

        // Argmax over the (beamSource * beamChild) dimension
        topK =
                newProbs.reshape(numBatch, numBeam * numBeam)
                        .topK(Math.toIntExact(numBeam), -1, true, false);

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
