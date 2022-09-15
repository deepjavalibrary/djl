/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.timeseries.block;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * A class computes a scaling factor as the weighted average absolute value along dimension {@code
 * dim}, and scales the data accordingly.
 */
public class MeanScaler extends Scaler {

    private float minimumScale;

    MeanScaler(Builder builder) {
        super(builder);
        minimumScale = builder.minimumScale;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray data = inputs.get(0);
        NDArray weights = inputs.get(1);

        NDArray totalWeight = weights.sum(new int[] {dim});
        NDArray weightedSum = data.abs().mul(weights).sum(new int[] {dim});

        NDArray totalObserved = totalWeight.sum(new int[] {0});
        NDArray denominator = NDArrays.maximum(totalObserved, 1f);
        NDArray defaultScale = weightedSum.sum(new int[] {0}).div(denominator);

        denominator = NDArrays.maximum(totalWeight, 1f);
        NDArray scale = weightedSum.div(denominator);

        scale =
                NDArrays.maximum(
                                minimumScale,
                                NDArrays.where(
                                        weightedSum.gt(weightedSum.zerosLike()),
                                        scale,
                                        defaultScale.mul(totalWeight.onesLike())))
                        .expandDims(dim);

        return new NDList(data.div(scale), keepDim ? scale : scale.squeeze(dim));
    }

    /**
     * Create a builder to build a {@code MeanScaler}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder to construct a {@code MeanScaler}. */
    public static final class Builder extends ScalerBuilder<Builder> {

        private float minimumScale = 1e-10f;

        Builder() {}

        /**
         * Sets the minimum scalar of the data.
         *
         * @param minimumScale the minimum value
         * @return this Builder
         */
        public Builder optMinimumScale(float minimumScale) {
            this.minimumScale = minimumScale;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Return the constructed {@code MeanScaler}.
         *
         * @return the constructed {@code MeanScaler}
         */
        public MeanScaler build() {
            validate();
            return new MeanScaler(this);
        }
    }
}
