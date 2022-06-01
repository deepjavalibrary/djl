/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.nn.norm;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.translate.Batchifier;
import ai.djl.translate.StackBatchifier;
import ai.djl.util.PairList;

/**
 * {@link GhostBatchNorm} is similar to {@link BatchNorm} except that it splits a batch into a
 * smaller sub-batches aka <em>ghost batches</em>, and normalize them individually to have a mean of
 * 0 and variance of 1 and finally concatenate them again to a single batch. Each of the
 * mini-batches contains a virtualBatchSize samples.
 *
 * @see <a href="https://arxiv.org/abs/1705.08741">Ghost Normalization Paper</a>
 */
public class GhostBatchNorm extends BatchNorm {

    private int virtualBatchSize;
    private Batchifier batchifier;

    protected GhostBatchNorm(Builder builder) {
        super(builder);

        this.virtualBatchSize = builder.virtualBatchSize;
        this.batchifier = new StackBatchifier();
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {

        NDList[] subBatches = split(inputs);
        for (NDList batch : subBatches) {
            super.forwardInternal(parameterStore, batch, training, params);
        }
        return batchify(subBatches);
    }

    /**
     * Splits an {@link NDList} into the given <b>size</b> of sub-batch.
     *
     * <p>This function unbatchifies the input {@link NDList} into mini-batches, each with the size
     * of virtualBatchSize. If the batch size is divisible by the virtual batch size, all returned
     * sub-batches will be the same size. If the batch size is not divisible by virtual batch size,
     * all returned sub-batches will be the same size, except the last one.
     *
     * @param list the {@link NDList} that needs to be split
     * @return an array of {@link NDList} that contains all the mini-batches
     */
    protected NDList[] split(NDList list) {
        double batchSize = list.head().size(0);
        int countBatches = (int) Math.ceil(batchSize / virtualBatchSize);

        return batchifier.split(list, countBatches, true);
    }

    /**
     * Converts an array of {@link NDList} into an NDList using {@link StackBatchifier} and squeezes
     * the first dimension created by it. This makes the final {@link NDArray} same size as the
     * splitted one
     *
     * @param subBatches the input array of {@link NDList}
     * @return the batchified {@link NDList}
     */
    protected NDList batchify(NDList[] subBatches) {
        NDList batch = batchifier.batchify(subBatches);

        return squeezeExtraDimensions(batch);
    }

    /**
     * Squeezes first axes of {@link NDList}
     *
     * @param batch input array of {@link NDList}
     * @return the squeezed {@link NDList}
     */
    protected NDList squeezeExtraDimensions(NDList batch) {
        NDArray array = batch.singletonOrThrow().squeeze(0);
        batch.set(0, array);

        return batch;
    }

    /**
     * Creates a builder to build a {@code GhostBatchNorm}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link GhostBatchNorm}. */
    public static class Builder extends BatchNorm.BaseBuilder<Builder> {
        private int virtualBatchSize = 128;

        Builder() {}

        /**
         * Set the size of virtual batches in which to use when sub-batching. Defaults to 128.
         *
         * @param virtualBatchSize the virtual batch size
         * @return this Builder
         */
        public Builder optVirtualBatchSize(int virtualBatchSize) {
            this.virtualBatchSize = virtualBatchSize;
            return this;
        }

        /**
         * Builds the new {@link GhostBatchNorm}.
         *
         * @return the new {@link GhostBatchNorm}
         */
        public GhostBatchNorm build() {
            return new GhostBatchNorm(this);
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }
    }
}
