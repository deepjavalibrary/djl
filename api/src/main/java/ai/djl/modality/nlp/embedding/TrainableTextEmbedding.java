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

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * {@code TrainableTextEmbedding} is an implementation of {@link TextEmbedding} based on {@link
 * TrainableWordEmbedding} block. This {@link TextEmbedding} is ideal when there are no pre-trained
 * embeddings available, or when the pre-trained embedding needs to be further trained.
 */
public class TrainableTextEmbedding extends AbstractBlock implements TextEmbedding {
    private static final byte VERSION = 1;
    private TrainableWordEmbedding trainableWordEmbedding;

    /**
     * Constructs a {@link SimpleTextEmbedding}.
     *
     * @param wordEmbedding the word embedding to embed each word
     */
    public TrainableTextEmbedding(TrainableWordEmbedding wordEmbedding) {
        this.trainableWordEmbedding = wordEmbedding;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray preprocessTextToEmbed(NDManager manager, List<String> text) {
        NDList result = new NDList(text.size());
        for (String token : text) {
            result.add(trainableWordEmbedding.preprocessWordToEmbed(manager, token));
        }
        return NDArrays.stack(result);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedText(NDArray text) {
        throw new UnsupportedOperationException("This operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public List<String> unembedText(NDArray textEmbedding) throws EmbeddingException {
        NDList split = textEmbedding.split(textEmbedding.getShape().get(0));
        List<String> result = new ArrayList<>(split.size());
        for (NDArray token : split) {
            result.add(trainableWordEmbedding.unembedWord(token.get(0)));
        }
        return result;
    }

    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return trainableWordEmbedding.forward(parameterStore, inputs, params);
    }

    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        beforeInitialize(inputShapes);
        return trainableWordEmbedding.initialize(manager, dataType, inputShapes);
    }

    @Override
    public BlockList getChildren() {
        return new BlockList(
                Collections.singletonList("trainableWordEmbedding"),
                Collections.singletonList(trainableWordEmbedding));
    }

    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("TrainableTextEmbedding have no parameters");
    }

    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return trainableWordEmbedding.getOutputShapes(manager, inputShapes);
    }

    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        os.writeByte(VERSION);
        saveInputShapes(os);
        trainableWordEmbedding.saveParameters(os);
    }

    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        byte version = is.readByte();
        if (version != VERSION) {
            throw new MalformedModelException("Unsupported encoding version: " + version);
        }
        readInputShapes(is);
        trainableWordEmbedding.loadParameters(manager, is);
    }
}
