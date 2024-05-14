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

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Embedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** An implement of nn.embedding. */
public final class FeatureEmbedding extends AbstractBlock {

    private static final String EMBEDDING_PARAM_NAME = "embedding";

    private int embeddingSize;
    private int numEmbeddings;

    private Parameter embedding;

    FeatureEmbedding(Builder builder) {
        embeddingSize = builder.embeddingSize;
        numEmbeddings = builder.numEmbeddings;
        embedding =
                addParameter(
                        Parameter.builder()
                                .setName(EMBEDDING_PARAM_NAME)
                                .setType(Parameter.Type.WEIGHT)
                                .optShape(new Shape(numEmbeddings, embeddingSize))
                                .build());
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray input = inputs.singletonOrThrow();
        Device device = input.getDevice();
        NDArray weight = parameterStore.getValue(embedding, device, training);
        return Embedding.embedding(input, weight, SparseFormat.DENSE);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[] {inputShapes[0].addAll(new Shape(embeddingSize))};
    }

    /**
     * Return a builder to build an {@code FeatureEmbedding}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** The builder to construct a {@link FeatureEmbedding} type of {@link ai.djl.nn.Block}. */
    public static final class Builder {

        private int embeddingSize;
        private int numEmbeddings;

        /**
         * Sets the size of the embeddings.
         *
         * @param embeddingSize the size of the embeddings
         * @return this Builder
         */
        public Builder setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets the size of the dictionary of embeddings.
         *
         * @param numEmbeddings the size of the dictionary of embeddings
         * @return this Builder
         */
        public Builder setNumEmbeddings(int numEmbeddings) {
            this.numEmbeddings = numEmbeddings;
            return this;
        }

        /**
         * Return the constructed {@code FeatureEmbedding}.
         *
         * @return the constructed {@code FeatureEmbedding}
         */
        public FeatureEmbedding build() {
            if (numEmbeddings <= 0) {
                throw new IllegalArgumentException(
                        "You must specify the dictionary Size for the embedding.");
            }
            if (embeddingSize == 0) {
                throw new IllegalArgumentException("You must specify the embedding size");
            }
            return new FeatureEmbedding(this);
        }
    }
}
