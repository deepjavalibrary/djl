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
/**
 * BatchTensorList represents a search state, and the NDArrays inside are updated in each iteration
 * of the autoregressive loop It is a struct consisting of NDArrays, whose first dimension is batch,
 * and also contains sequence dimension (whose position in tensor's shape is specified by seqDimOrder).
 * The SeqBatcher batch operations will operate on these two dimensions.
 */
public abstract class BatchTensorList {
    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastOutputIds;

    // [batch, seq_past]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastAttentionMask;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDList pastKeyValues;

    // Sequence dimension order among all dimensions for each element in the batch list.
    private long[] seqDimOrder;

    BatchTensorList() {}

    /**
     * Constructs a BatchTensorList.
     *
     * @param list the NDList that contains the serialized version of the batch tensors
     * @param seqDimOrder the sequence dimension order that specifies where the sequence dimension
     *     is in a tensor's shape
     */
    BatchTensorList(NDList list, long[] seqDimOrder) {
        this.seqDimOrder = seqDimOrder;
        pastOutputIds = list.get(0);
        pastAttentionMask = list.get(1);
        pastKeyValues = list.subNDList(2);
    }

    /**
     * Constructs a BatchTensorList.
     *
     * @param pastOutputIds past output token ids
     * @param pastAttentionMask past attention mask
     * @param pastKeyValues past kv cache
     * @param seqDimOrder the sequence dimension order that specifies where the sequence dimension
     *     is in a tensor's shape
     */
    BatchTensorList(
            NDArray pastOutputIds,
            NDArray pastAttentionMask,
            NDList pastKeyValues,
            long[] seqDimOrder) {
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
        this.seqDimOrder = seqDimOrder;
    }

    /**
     * Construct a BatchTensorList from the serialized version of the batch tensors. The
     * pastOutputIds has to be the first in the output list.
     *
     * @param inputList the serialized version of the batch tensors
     * @param seqDimOrder the sequence dimension order that specifies where the sequence dimension
     *     is in a tensor's shape
     * @return BatchTensorList
     */
    public abstract BatchTensorList fromList(NDList inputList, long[] seqDimOrder);

    /**
     * Gets the serialized version of the BatchTensorList. The pastOutputIds has to be the first in
     * the output list.
     *
     * @return the NDList that contains the serialized BatchTensorList
     */
    public abstract NDList getList();

    /**
     * Gets the sequence dimension order which specifies where the sequence dimension is in a
     * tensor's shape.
     *
     * @return the sequence dimension order which specifies where the sequence dimension is in a
     *     tensor's shape
     */
    public long[] getSeqDimOrder() {
        return seqDimOrder;
    }

    /**
     * Gets the value of the pastOutputIds.
     *
     * @return the value of pastOutputIds
     */
    public NDArray getPastOutputIds() {
        return pastOutputIds;
    }

    /**
     * Sets the past output token ids.
     *
     * @param pastOutputIds the past output token ids
     */
    public void setPastOutputIds(NDArray pastOutputIds) {
        this.pastOutputIds = pastOutputIds;
    }

    /**
     * Gets the value of the pastAttentionMask.
     *
     * @return the value of pastAttentionMask
     */
    public NDArray getPastAttentionMask() {
        return pastAttentionMask;
    }

    /**
     * Sets the attention mask.
     *
     * @param pastAttentionMask the attention mask
     */
    public void setPastAttentionMask(NDArray pastAttentionMask) {
        this.pastAttentionMask = pastAttentionMask;
    }

    /**
     * Gets the value of the pastKeyValues.
     *
     * @return the value of pastKeyValues
     */
    public NDList getPastKeyValues() {
        return pastKeyValues;
    }

    /**
     * Sets the kv cache.
     *
     * @param pastKeyValues the kv cache
     */
    public void setPastKeyValues(NDList pastKeyValues) {
        this.pastKeyValues = pastKeyValues;
    }

    /**
     * Sets the sequence dimension order which specifies where the sequence dimension is in a
     * tensor's shape.
     *
     * @param seqDimOrder the sequence dimension order which specifies where the sequence dimension
     *     is in a tensor's shape
     */
    public void setSeqDimOrder(long[] seqDimOrder) {
        this.seqDimOrder = seqDimOrder;
    }
}
