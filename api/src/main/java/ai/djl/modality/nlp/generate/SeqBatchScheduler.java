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
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * This is a scheduler, serving as an API to the consumer of the system, allowing for three major
 * actions: initForward, addBatch, fastForward, collectResults. An optimal control sequence should
 * be solved, after considering the time consumption of each action, the batch size and sequence
 * length of queueing requests. Such optimal control solver needs additional effort. Primitive
 * policy is setting several thresholds.
 */
public abstract class SeqBatchScheduler {
    private static final Logger logger = LoggerFactory.getLogger(SeqBatchScheduler.class);

    Predictor<NDList, CausalLMOutput> predictor;
    SeqBatcher seqBatcher;

    NDManager manager;

    SearchConfig config;

    Map<Long, NDArray> results;

    /**
     * Constructs a new {@code SeqBatchScheduler} instance.
     *
     * @param lmBlock the predictor that cont
     * @param config the search parameter configuration
     */
    public SeqBatchScheduler(Predictor<NDList, CausalLMOutput> lmBlock, SearchConfig config) {
        this.predictor = lmBlock;
        this.config = config;
        results = new ConcurrentHashMap<>();
    }

    /**
     * Initializes the iteration and SeqBatcher.
     *
     * @param inputIds the input token ids.
     * @param batchUids the request uid identifying a sequence
     * @return SeqBatcher Stores the search state and operate on the BatchTensorList
     * @throws TranslateException if forward fails
     */
    public abstract SeqBatcher initForward(NDArray inputIds, NDArray batchUids)
            throws TranslateException;

    /**
     * Executes forward for a given number of iterations.
     *
     * @param count the time of forward calls
     * @return boolean Indicate whether the Batch is empty
     * @throws TranslateException if forward fails
     */
    public boolean incrementForward(int count) throws TranslateException {
        int i = 0;
        while (i++ < count) {
            if (seqBatcher == null || seqBatcher.getData() == null) {
                logger.info(
                        "seqBatcher not set or is empty. Please call addBatch. Current inference"
                                + " times is "
                                + i);
                return true;
            }

            inferenceCall();
            if (seqBatcher.sequenceComplete()) {
                results.putAll(seqBatcher.collectAndTrim());
            }
        }
        return false;
    }

    /**
     * An inference call in an iteration.
     *
     * @return the output token ids
     * @throws TranslateException if forward fails
     */
    abstract NDArray inferenceCall() throws TranslateException;

    /**
     * Adds new batch.
     *
     * @param inputIds the input token ids.
     * @param batchUids the request uid identifying a sequence
     * @throws TranslateException if forward fails
     */
    public void addRequest(NDArray inputIds, NDArray batchUids) throws TranslateException {
        SeqBatcher seqBatcherNew = initForward(inputIds, batchUids);
        if (seqBatcher == null) {
            seqBatcher = seqBatcherNew;
        } else {
            seqBatcher.addBatch(seqBatcherNew);
        }
    }

    /**
     * Collects finished results.
     *
     * @return the outputs stored as a map from requestUid to output token ids
     */
    public Map<Long, NDArray> collectResults() {
        Map<Long, NDArray> output = results;
        results = new ConcurrentHashMap<>();
        return output;
    }

    /**
     * Computes the offSets by linear search from the left.
     *
     * @param inputIds input token ids
     * @param config search configuration
     * @return the offsets NDArray
     */
    static NDArray computeOffSets(NDArray inputIds, SearchConfig config) {
        int numBatch = Math.toIntExact(inputIds.getShape().get(0));
        int initSeqSize = Math.toIntExact(inputIds.getShape().get(1));

        // Linear search from left to find the first position that's not padTokenId.
        long[] offSetsArray = new long[numBatch];
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (aSequence[idx] != config.getPadTokenId()) {
                    break;
                }
                idx++;
            }
            offSetsArray[i] = idx;
        }

        NDManager manager = inputIds.getManager();
        return manager.create(offSetsArray).reshape(-1, 1);
    }

    /**
     * Computes the attention mask by linear search from the left.
     *
     * @param inputIds input token ids
     * @param config search configuration
     * @return the attention mask NDArray
     */
    static NDArray computeAttentionMask(NDArray inputIds, SearchConfig config) {
        int numBatch = Math.toIntExact(inputIds.getShape().get(0));
        int initSeqSize = Math.toIntExact(inputIds.getShape().get(1));

        NDManager manager = inputIds.getManager();
        NDArray attentionMask =
                manager.ones(new Shape(1, inputIds.getShape().getLastDimension()), DataType.INT64)
                        .reshape(1, -1)
                        .repeat(0, numBatch);

        // Linear search to find the offset and set the mask
        for (int i = 0; i < numBatch; i++) {
            long[] aSequence = inputIds.get("{},:", i).toLongArray();
            int idx = 0;
            while (idx < initSeqSize) {
                if (aSequence[idx] != config.getPadTokenId()) {
                    break;
                }
                idx++;
            }
            attentionMask.set(new NDIndex("{},{}:{}", i, 0, idx), 0);
        }

        // [batch, pastSeq]
        return attentionMask;
    }

    /**
     * Computes the position ids by linear search from the left.
     *
     * @param inputIds input token ids
     * @param offSets the offset
     * @param pastSeqLength past sequence length
     * @param repeat the number of repeats used in interleave-repeating the position_ids to multiple
     *     rows
     * @return the position ids NDArray
     */
    static NDArray computePositionIds(
            NDArray inputIds, NDArray offSets, long pastSeqLength, int repeat) {
        NDManager manager = inputIds.getManager();
        NDArray positionIds =
                manager.arange(
                                pastSeqLength,
                                pastSeqLength + inputIds.getShape().getLastDimension(),
                                1,
                                DataType.INT64)
                        .expandDims(0)
                        .repeat(0, inputIds.getShape().get(0));

        NDArray positionIdsShifted = positionIds.subi(offSets.reshape(-1, 1).repeat(0, repeat));
        positionIds = positionIdsShifted.maximum(positionIdsShifted.zerosLike());

        // [batch, inputSeq]
        return positionIds;
    }
}
