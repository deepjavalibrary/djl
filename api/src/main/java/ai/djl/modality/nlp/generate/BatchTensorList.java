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

// BatchTensorList represents a search state, and the NDArrays inside are updated in each iteration
// of the
// autoregressive loop.
// It is a struct consisting of NDArrays, whose first dimension is batch, and also contains
// sequence dimension (whose position in tensor's shape is specified by seqDimOrder).
// The SeqBatcher batch operations will operate on these two dimensions.
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

    BatchTensorList(NDList list, long[] seqDimOrder) {
        this.seqDimOrder = seqDimOrder;
        pastOutputIds = list.get(0);
        pastAttentionMask = list.get(1);
        pastKeyValues = list.subNDList(2);
    }

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

    public abstract BatchTensorList fromList(NDList inputList, long[] seqDimOrder);

    // The pastOutputIds has to be the first in the output list
    public abstract NDList getList();

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

    public void setPastKeyValues(NDList pastKeyValues) {
        this.pastKeyValues = pastKeyValues;
    }

    public void setSeqDimOrder(long[] seqDimOrder) {
        this.seqDimOrder = seqDimOrder;
    }
}
