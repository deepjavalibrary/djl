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
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

import java.util.ArrayList;
import java.util.List;

/** Embed a sequence of categorical features. */
public class FeatureEmbedder extends AbstractBlock {

    private List<Integer> cardinalities;
    private List<Integer> embeddingDims;
    private List<FeatureEmbedding> embedders;
    private int numFeatures;

    FeatureEmbedder(Builder builder) {
        cardinalities = builder.cardinalities;
        embeddingDims = builder.embeddingDims;
        numFeatures = cardinalities.size();
        embedders = new ArrayList<>();
        for (int i = 0; i < cardinalities.size(); i++) {
            embedders.add(createEmbedding(i, cardinalities.get(i), embeddingDims.get(i)));
        }
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        // Categorical features with shape: (N,T,C) or (N,C), where C is the number of categorical
        // features.
        NDArray features = inputs.singletonOrThrow();

        NDList catFeatureSlices;
        if (numFeatures > 1) {
            // slice the last dimension, giving an array of length numFeatures with shape (N,T) or
            // (N)
            catFeatureSlices = features.split(numFeatures, features.getShape().dimension() - 1);
        } else {
            catFeatureSlices = new NDList(features);
        }

        NDList output = new NDList();
        for (int i = 0; i < numFeatures; i++) {
            FeatureEmbedding embed = embedders.get(i);
            NDArray catFeatureSlice = catFeatureSlices.get(i);
            catFeatureSlice = catFeatureSlice.squeeze(-1);
            output.add(
                    embed.forward(parameterStore, new NDList(catFeatureSlice), training, params)
                            .singletonOrThrow());
        }
        return new NDList(NDArrays.concat(output, -1));
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        Shape inputShape = inputShapes[0];
        Shape[] embedInputShapes = {inputShape.slice(0, inputShape.dimension() - 1)};
        long embedSizes = 0;
        for (FeatureEmbedding embed : embedders) {
            embedSizes += embed.getOutputShapes(embedInputShapes)[0].tail();
        }
        return new Shape[] {inputShape.slice(0, inputShape.dimension() - 1).add(embedSizes)};
    }

    /** {@inheritDoc} */
    @Override
    protected void initializeChildBlocks(
            NDManager manager, DataType dataType, Shape... inputShapes) {
        for (FeatureEmbedding embed : embedders) {
            embed.initialize(manager, dataType, inputShapes);
        }
    }

    private FeatureEmbedding createEmbedding(int i, int c, int d) {
        FeatureEmbedding embedding =
                FeatureEmbedding.builder().setNumEmbeddings(c).setEmbeddingSize(d).build();
        addChildBlock(String.format("cat_%d_embedding", i), embedding);
        return embedding;
    }

    /**
     * Return a builder to build an {@code FeatureEmbedder}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder to construct a {@link FeatureEmbedder} type of {@link ai.djl.nn.Block}. */
    public static final class Builder {

        private List<Integer> cardinalities;
        private List<Integer> embeddingDims;

        /**
         * Set the cardinality for each categorical feature.
         *
         * @param cardinalities the cardinality for each categorical feature
         * @return this Builder
         */
        public Builder setCardinalities(List<Integer> cardinalities) {
            this.cardinalities = cardinalities;
            return this;
        }

        /**
         * Set the number of dimensions to embed each categorical feature.
         *
         * @param embeddingDims number of dimensions to embed each categorical feature
         * @return this Builder
         */
        public Builder setEmbeddingDims(List<Integer> embeddingDims) {
            this.embeddingDims = embeddingDims;
            return this;
        }

        /**
         * Return the constructed {@code FeatureEmbedder}.
         *
         * @return the constructed {@code FeatureEmbedder}
         */
        public FeatureEmbedder build() {
            if (cardinalities.isEmpty()) {
                throw new IllegalArgumentException(
                        "Length of 'cardinalities' list must be greater than zero");
            }
            if (cardinalities.size() != embeddingDims.size()) {
                throw new IllegalArgumentException(
                        "Length of `cardinalities` and `embedding_dims` should match");
            }
            for (int c : cardinalities) {
                if (c <= 0) {
                    throw new IllegalArgumentException("Elements of `cardinalities` should be > 0");
                }
            }
            for (int d : embeddingDims) {
                if (d <= 0) {
                    throw new IllegalArgumentException(
                            "Elements of `embedding_dims` should be > 0");
                }
            }
            return new FeatureEmbedder(this);
        }
    }
}
