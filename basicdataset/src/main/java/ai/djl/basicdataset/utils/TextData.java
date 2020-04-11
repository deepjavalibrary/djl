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
package ai.djl.basicdataset.utils;

import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.SimpleTextEmbedding;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * {@link TextData} is a utility for managing textual data within a {@link
 * ai.djl.training.dataset.Dataset}.
 *
 * <p>See {@link ai.djl.basicdataset.TextDataset} for an example.
 */
public class TextData {

    private List<TextProcessor> textProcessors;
    private TextEmbedding textEmbedding;
    private boolean trainEmbedding;
    private int embeddingSize;

    private List<List<String>> textData;
    private int size;

    /**
     * Embds the text at a given index to an NDList.
     *
     * <p>Follows an embedding strategy based on {@link #trainEmbedding}.
     *
     * @param index the index of the data to embed
     * @param manager the manager for the embedded array
     * @return the embedded array
     * @throws EmbeddingException if the value could not be embedded
     */
    public NDList embedText(long index, NDManager manager) throws EmbeddingException {
        int iindex = Math.toIntExact(index);
        NDList data = new NDList();

        List<String> sentenceTokens = textData.get(iindex);
        if (trainEmbedding) {
            data.add(textEmbedding.preprocessTextToEmbed(manager, sentenceTokens));
        } else {
            data.add(textEmbedding.embedText(manager, sentenceTokens));
        }
        return data;
    }

    /**
     * Preprocess the textData by providing the data from the dataset.
     *
     * @param newTextData the data from the dataset
     */
    public void preprocess(List<String> newTextData) {
        SimpleVocabulary.VocabularyBuilder vocabularyBuilder =
                new SimpleVocabulary.VocabularyBuilder();
        vocabularyBuilder.optMinFrequency(3);
        vocabularyBuilder.optReservedTokens(Arrays.asList("<pad>", "<bos>", "<eos>"));

        if (textData == null) {
            textData = new ArrayList<>();
        }
        size = textData.size();
        for (String textDatum : newTextData) {
            List<String> tokens = Collections.singletonList(textDatum);
            for (TextProcessor processor : textProcessors) {
                tokens = processor.preprocess(tokens);
            }
            vocabularyBuilder.add(tokens);
            textData.add(tokens);
        }
        SimpleVocabulary vocabulary = vocabularyBuilder.build();
        for (int i = 0; i < textData.size(); i++) {
            List<String> tokenizedTextDatum = textData.get(i);
            for (int j = 0; j < tokenizedTextDatum.size(); j++) {
                if (!vocabulary.isKnownToken(tokenizedTextDatum.get(j))) {
                    tokenizedTextDatum.set(j, vocabulary.getUnknownToken());
                }
            }
            textData.set(i, tokenizedTextDatum);
        }
        if (textEmbedding == null) {
            textEmbedding =
                    new SimpleTextEmbedding(new TrainableWordEmbedding(vocabulary, embeddingSize));
            trainEmbedding = true;
        }
    }

    /**
     * Sets the text processors.
     *
     * @param textProcessors the new textProcessors
     */
    public void setTextProcessors(List<TextProcessor> textProcessors) {
        this.textProcessors = textProcessors;
    }

    /**
     * Sets the textEmbedding to embed the data with.
     *
     * @param textEmbedding the textEmbedding
     */
    public void setTextEmbedding(TextEmbedding textEmbedding) {
        this.textEmbedding = textEmbedding;
    }

    /**
     * Sets whether to train the textEmbedding.
     *
     * @param trainEmbedding true to train the text embedding
     */
    public void setTrainEmbedding(boolean trainEmbedding) {
        this.trainEmbedding = trainEmbedding;
    }

    /**
     * Sets the default embedding size.
     *
     * @param embeddingSize the default embedding size
     */
    public void setEmbeddingSize(int embeddingSize) {
        this.embeddingSize = embeddingSize;
    }

    /**
     * Returns the size of the data.
     *
     * @return the size of the data
     */
    public int getSize() {
        return size;
    }
}
