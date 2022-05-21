package ai.djl.nn.norm;

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
        for (NDList batch : subBatches)
            super.forwardInternal(parameterStore, batch, training, params);

        return batchifier.batchify(subBatches);
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
     * Creates a builder to build a {@code GhostBatchNorm.}
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The Builder to construct a {@link GhostBatchNorm} */
    public static class Builder extends BatchNorm.BaseBuilder<Builder> {
        private int virtualBatchSize = 128;

        Builder() {}

        /**
         * Set the size of virtual batches in which to use when sub-batching. Defaults to 128.
         *
         * @param virtualBatchSize
         * @return this Builder
         */
        public Builder optVirtualBatchSize(int virtualBatchSize) {
            this.virtualBatchSize = virtualBatchSize;
            return this;
        }

        /**
         * Builds the new {@link BatchNorm}.
         *
         * @return the new {@link BatchNorm}
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
