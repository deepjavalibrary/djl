/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.modality.nlp.embedding;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code TrainableTextEmbedding} is an implementation of {@link TextEmbedding} based on {@link
 * TrainableWordEmbedding} block. This {@link TextEmbedding} is ideal when there are no pre-trained
 * embeddings available, or when the pre-trained embedding needs to be further trained.
 */
public class TrainableTextEmbedding extends AbstractBlock implements TextEmbedding {

    private TrainableWordEmbedding trainableWordEmbedding;

    /**
     * Constructs a {@link TrainableTextEmbedding}.
     *
     * @param wordEmbedding the word embedding to embed each word
     */
    public TrainableTextEmbedding(TrainableWordEmbedding wordEmbedding) {
        this.trainableWordEmbedding = addChildBlock("trainableWordEmbedding", wordEmbedding);
    }

    /** {@inheritDoc} */
    @Override
    public long[] preprocessTextToEmbed(List<String> text) {
        long[] result = new long[text.size()];
        for (int i = 0; i < text.size(); i++) {
            result[i] = trainableWordEmbedding.preprocessWordToEmbed(text.get(i));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray textIndices) throws EmbeddingException {
        throw new UnsupportedOperationException(
                "EmbedText operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray textEmbedding) {
        NDList split = textEmbedding.split(textEmbedding.getShape().get(0));
        List<String> result = new ArrayList<>(split.size());
        for (NDArray token : split) {
            result.add(trainableWordEmbedding.unembedWord(token.get(0)));
        }
        return result;
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        return trainableWordEmbedding.forward(parameterStore, inputs, training, params);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        trainableWordEmbedding.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return trainableWordEmbedding.getOutputShapes(inputShapes);
    }
}
