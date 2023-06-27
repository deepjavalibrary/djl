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

class GreedyBatchTensorList extends BatchTensorList {
    // [batch, 1]
    private NDArray nextInputIds;

    // [batch, seq_past + new_seq]
    // The cache of past attentionMask. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastAttentionMask;

    /* Variables below are one time step behind the above state variables. Ie, they contain all the past sequence but excludes the time step that corresponds to the above input. */

    // [batch, seq_past]. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDArray pastOutputIds;

    // (k, v) * numLayer,
    // kv: [batch, heads, seq_past, kvfeature]
    // The cache of past sequence. seq-dim-size == |past_seq| + |inputIds|. Will grow.
    private NDList pastKeyValues;

    GreedyBatchTensorList(
            NDArray nextInputIds,
            NDArray pastOutputIds,
            NDList pastKeyValues,
            NDArray pastAttentionMask) {
        this.nextInputIds = nextInputIds;
        this.pastKeyValues = pastKeyValues;
        this.pastOutputIds = pastOutputIds;
        this.pastAttentionMask = pastAttentionMask;
    }

    public GreedyBatchTensorList() {}

    /** {@inheritDoc} */
    @Override
    public BatchTensorList fromList(NDList inputList, long[] seqDimOrder) {
        return new GreedyBatchTensorList();
    }

    /** {@inheritDoc} */
    @Override
    public NDList getList() {
        return new NDList();
    }

    /**
     * Returns the value of the nextInputIds.
     *
     * @return the value of nextInputIds
     */
    public NDArray getNextInputIds() {
        return nextInputIds;
    }

    public void setNextInputIds(NDArray nextInputIds) {
        this.nextInputIds = nextInputIds;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getPastAttentionMask() {
        return pastAttentionMask;
    }

    /** {@inheritDoc} */
    @Override
    public void setPastAttentionMask(NDArray pastAttentionMask) {
        this.pastAttentionMask = pastAttentionMask;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getPastOutputIds() {
        return pastOutputIds;
    }

    /** {@inheritDoc} */
    @Override
    public void setPastOutputIds(NDArray pastOutputIds) {
        this.pastOutputIds = pastOutputIds;
    }

    /** {@inheritDoc} */
    @Override
    public NDList getPastKeyValues() {
        return pastKeyValues;
    }

    /** {@inheritDoc} */
    @Override
    public void setPastKeyValues(NDList pastKeyValues) {
        this.pastKeyValues = pastKeyValues;
    }
}
