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
import ai.djl.ndarray.types.Shape;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code SeqBatcher} stores the search state (BatchTensorList), the control variables (e.g.
 * seqLength, offSets, etc), and batch operations (merge, trim, exitCriteria, etc) on
 * BatchTensorList.
 */
public class SeqBatcher {

    NDManager manager;
    long batchSize;
    long seqLength;

    /** [batch] stores the uid and is trimmed or enhanced correspondingly to the batch. */
    NDArray batchUid;

    /** Minheap with lazy removal, map: batchIdx -> offset. */
    NDArray offSets;

    /** This is a struct that contains NDArrays with batch dimension. */
    BatchTensorList data;

    /** batchIndex -> seqEndPosition. */
    private Map<Long, Long> exitIndexEndPosition;

    SeqBatcher(BatchTensorList data, NDArray batchUid, NDArray offSets, NDManager manager) {
        this.manager = manager.newSubManager();
        this.data = data;
        this.batchUid = batchUid.getShape().dimension() == 2 ? batchUid : batchUid.reshape(-1, 1);
        this.offSets = offSets.getShape().hashCode() == 2 ? offSets : offSets.reshape(-1, 1);
        batchSize = data.getPastOutputIds().getShape().get(0);
        seqLength = data.getPastOutputIds().getShape().get(1);
        exitIndexEndPosition = new ConcurrentHashMap<>();
    }

    /**
     * Returns the batch data which is stored as a {@code BatchTensorList}.
     *
     * @return the batch data stored as BatchTensorList
     */
    public BatchTensorList getData() {
        return data;
    }

    /**
     * Adds new batch.
     *
     * <p>Modify the batch dimension and the left padding.
     *
     * @param seqBatcherNew the seqBatcher to add.
     */
    public void addBatch(SeqBatcher seqBatcherNew) {
        merge(this, seqBatcherNew, seqLength - seqBatcherNew.seqLength);
        // manager and finishedSequences stay the same;
    }

    /**
     * Merges two batchers together.
     *
     * <p>Modify the batch dimension and the left padding.
     *
     * @param seqBatcher1 the first seqBatcher
     * @param seqBatcher2 the second seqBatcher
     * @param seqDelta the sequence length difference
     */
    private void merge(SeqBatcher seqBatcher1, SeqBatcher seqBatcher2, long seqDelta) {
        if (seqDelta < 0) {
            SeqBatcher swapTmp = seqBatcher1;
            seqBatcher1 = seqBatcher2;
            seqBatcher2 = swapTmp;
            seqDelta = -seqDelta;
        }

        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();

            NDList list1 = seqBatcher1.data.getList();
            NDList list2 = seqBatcher2.data.getList();
            NDList merged = new NDList(list1.size());
            long[] seqDimOrder = seqBatcher1.data.getSeqDimOrder();
            for (int i = 0; i < list1.size(); i++) {
                NDArray batch1 = list1.get(i);
                NDArray batch2 = list2.get(i);
                if (seqDelta == 0) {
                    // no need to pad
                    batch1 = batch1.concat(batch2, 0);
                    merged.add(batch1);
                    continue;
                }

                long[] shape1 = batch1.getShape().getShape();
                long[] shape2 = batch2.getShape().getShape();
                long padTokenId = 220;

                // Augment the larger, batch1
                long[] shapeDelta = batch1.getShape().getShape();
                shapeDelta[0] = shape2[0];
                NDArray deltaArray;
                if (i == 0) {
                    // The outputTokenIds is padded with padTokenId
                    deltaArray =
                            manager.full(new Shape(shapeDelta), padTokenId, batch1.getDataType());
                } else {
                    // The rest e.g. attentionMask, kvCache, hiddenStates are padded with 0
                    deltaArray = manager.zeros(new Shape(shapeDelta), batch1.getDataType());
                }
                batch1 = batch1.concat(deltaArray, 0);

                // Get the ndIndex used to set the extended part of batch1 to be batch2.
                NDIndex ndIndex;
                // Find the ordinal number of the sequence dimension
                if (seqDimOrder[i] > 0) {
                    // Has a valid sequence dimension
                    ndIndex = new NDIndex("{}:", seqBatcher1.batchSize);
                    int order = 1;
                    while (order < seqDimOrder[i]) {
                        ndIndex = ndIndex.addAllDim();
                        order++;
                    }
                    assert seqDelta + shape2[order] == shape1[order]
                            : "Wrong shapes. batch1 and batch2 are not mergable";
                    ndIndex = ndIndex.addSliceDim(seqDelta, shape1[order]).addEllipseDim();
                } else {
                    // Only batch dimension, no valid sequence dimension
                    ndIndex = new NDIndex("{}:, ...", seqBatcher1.batchSize);
                }

                // Copy batch2 to the extended part in batch1
                batch1.set(ndIndex, batch2);
                merged.add(batch1);
            }
            data = data.fromList(merged, data.getSeqDimOrder());

            batchSize = seqBatcher1.batchSize + seqBatcher2.batchSize;
            batchUid = seqBatcher1.batchUid.concat(seqBatcher2.batchUid, 0);
            offSets = seqBatcher1.offSets.concat(seqBatcher2.offSets.addi(seqDelta), 0);
            seqLength = seqBatcher1.seqLength;

            // memory
            NDScope.unregister(batchUid, offSets);
            NDScope.unregister(merged);
        }
    }

    /**
     * Checks which batch needs to exit, according certain criteria like EOS or maxLength.
     *
     * <p>It is an iteration over batch and is thus also considered as batch operation.
     *
     * @param outputIds output token ids in an incremental forward call
     * @param maxLength max total sequence length
     * @param eosTokenId end of sentence token id
     */
    public void exitCriteria(NDArray outputIds, long maxLength, long eosTokenId) {
        long[] outputIdsArray = outputIds.toLongArray();
        long[] offSetsArray = offSets.toLongArray();
        for (int i = 0; i < outputIdsArray.length; i++) {
            if (seqLength - offSetsArray[i] >= maxLength || outputIdsArray[i] == eosTokenId) {
                if (!exitIndexEndPosition.containsKey((long) i)) {
                    exitIndexEndPosition.put((long) i, seqLength);
                }
            }
        }
    }

    /**
     * Collects the finished sequences and trim the left padding.
     *
     * @return a map that stores request id to output token ids
     */
    public Map<Long, NDArray> collectAndTrim() {
        if (exitIndexEndPosition.isEmpty()) {
            return new ConcurrentHashMap<>();
        }
        Map<Long, NDArray> finishedSequences = new ConcurrentHashMap<>();

        try (NDScope scope = new NDScope()) {
            scope.suppressNotUsedWarning();
            // Collect the results into finishedSequences
            Set<Long> exitIndices = new HashSet<>();
            for (Map.Entry<Long, Long> entry : exitIndexEndPosition.entrySet()) {
                // batchIndex -> seqEndPosition
                long batchIndex = entry.getKey();
                long seqEndPosition = entry.getValue();
                long uid = batchUid.getLong(batchIndex);
                long offSet = offSets.getLong(batchIndex);
                NDArray output =
                        data.getPastOutputIds()
                                .get("{}, {}:{}", batchIndex, offSet, seqEndPosition);
                finishedSequences.put(uid, output);
                exitIndices.add(batchIndex);

                NDScope.unregister(output);
            }

            // Find the batch indices of the non-finished sequences.
            long[] keepIndices = new long[Math.toIntExact(batchSize) - exitIndices.size()];
            int j = 0;
            for (long i = 0; i < batchSize; i++) {
                if (!exitIndices.contains(i)) {
                    keepIndices[j++] = i;
                }
            }

            if (keepIndices.length == 0) {
                batchUid = manager.create(new Shape(0, 1), batchUid.getDataType());
                offSets = manager.create(new Shape(0, 1), offSets.getDataType());
                data = null;
                batchSize = 0;
                seqLength = 0;
                exitIndexEndPosition = new ConcurrentHashMap<>();

                NDScope.unregister(batchUid, offSets);
                return finishedSequences;
            }

            NDIndex ndIndex = new NDIndex("{}", manager.create(keepIndices));
            batchUid = batchUid.get(ndIndex).reshape(-1, 1);
            offSets = offSets.get(ndIndex).reshape(-1, 1);
            long trimSeq = offSets.min(new int[] {0}).toLongArray()[0];
            offSets = offSets.subi(trimSeq);

            // Trim batch, and sequence dim if needed
            NDList list = data.getList();
            NDList newList = new NDList(list.size());
            long[] seqDimOrder = data.getSeqDimOrder();
            for (int i = 0; i < list.size(); i++) {
                NDArray batch = list.get(i);
                if (trimSeq == 0) {
                    // no need to trim
                    ndIndex = new NDIndex("{}, ...", manager.create(keepIndices));
                    newList.add(batch.get(ndIndex));
                    continue;
                }

                // Get the ndIndex used to keep the entries and trim the rest
                // Find the ordinal number of the sequence dimension
                if (seqDimOrder[i] > 0) {
                    // Has a valid sequence dimension
                    ndIndex = new NDIndex("{}", manager.create(keepIndices));
                    int order = 1;
                    while (order < seqDimOrder[i]) {
                        ndIndex = ndIndex.addAllDim();
                        order++;
                    }
                    ndIndex = ndIndex.addSliceDim(trimSeq, seqLength).addEllipseDim();
                } else {
                    // Only batch dimension, no valid sequence dimension
                    ndIndex = new NDIndex("{}, ...", manager.create(keepIndices));
                }
                // Keep the indexed entries and trim the rest
                newList.add(batch.get(ndIndex));
            }
            data = data.fromList(newList, data.getSeqDimOrder());

            batchSize -= exitIndexEndPosition.size();
            seqLength -= trimSeq;

            exitIndexEndPosition = new ConcurrentHashMap<>();

            // memory
            NDScope.unregister(newList);
            NDScope.unregister(batchUid, offSets);

            return finishedSequences;
        }
    }

    /**
     * Computes the position ids by linear search from the left.
     *
     * @return the boolean indicating whether all sequences are empty
     */
    public boolean sequenceComplete() {
        return !exitIndexEndPosition.isEmpty();
    }
}
