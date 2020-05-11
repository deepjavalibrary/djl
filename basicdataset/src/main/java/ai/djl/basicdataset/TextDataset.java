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
package ai.djl.basicdataset;

import ai.djl.basicdataset.utils.TextData;
import ai.djl.basicdataset.utils.TextData.Configuration;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import java.util.List;

/**
 * {@code TextDataset} is an abstract dataset that can be used for datasets for natural language
 * processing where either the source or target are text-based data.
 *
 * <p>The {@code TextDataset} fetches the data in the form of {@link String}, processes the data as
 * required, and creates embeddings for the tokens. Embeddings can be either pre-trained or trained
 * on the go. Pre-trained {@link TextEmbedding} must be set in the {@link Builder}. If no embeddings
 * are set, the dataset creates {@link TrainableWordEmbedding} based {@link TrainableWordEmbedding}
 * from the {@link Vocabulary} created within the dataset.
 */
public abstract class TextDataset extends RandomAccessDataset {

    protected TextData sourceTextData;
    protected TextData targetTextData;
    protected NDManager manager;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public TextDataset(Builder<?> builder) {
        super(builder);
        sourceTextData =
                new TextData(
                        TextData.getDefaultConfiguration().update(builder.sourceConfiguration));
        targetTextData =
                new TextData(
                        TextData.getDefaultConfiguration().update(builder.targetConfiguration));
        manager = builder.manager;
    }

    /**
     * Gets the word embedding used while pre-processing the dataset. This method must be called
     * after preprocess has been called on this instance.
     *
     * @param source whether to get source or target text embedding
     * @return the text embedding
     */
    public TextEmbedding getTextEmbedding(boolean source) {
        TextData textData = source ? sourceTextData : targetTextData;
        return textData.getTextEmbedding();
    }

    /**
     * Gets the {@link SimpleVocabulary} built while preprocessing the text data.
     *
     * @param source whether to get source or target vocabulary
     * @return the {@link SimpleVocabulary}
     */
    public SimpleVocabulary getVocabulary(boolean source) {
        TextData textData = source ? sourceTextData : targetTextData;
        return textData.getVocabulary();
    }

    /**
     * Gets the raw textual input.
     *
     * @param index the index of the text input
     * @param source whether to get text from source or target
     * @return the raw text
     */
    public String getRawText(long index, boolean source) {
        TextData textData = source ? sourceTextData : targetTextData;
        return textData.getRawText(index);
    }

    /**
     * Performs pre-processing steps on text data such as tokenising, applying {@link
     * ai.djl.modality.nlp.preprocess.TextProcessor}s, creating vocabulary, and word embeddings.
     *
     * @param newTextData list of all unprocessed sentences in the dataset
     * @param source whether the text data provided is source or target
     * @throws EmbeddingException if there is an error while embedding
     */
    protected void preprocess(List<String> newTextData, boolean source) throws EmbeddingException {
        TextData textData = source ? sourceTextData : targetTextData;
        textData.preprocess(
                manager, newTextData.subList(0, (int) Math.min(limit, newTextData.size())));
    }

    /** Abstract Builder that helps build a {@link TextDataset}. */
    public abstract static class Builder<T extends Builder<T>> extends BaseBuilder<T> {
        private TextData.Configuration sourceConfiguration = new Configuration();
        private TextData.Configuration targetConfiguration = new Configuration();
        private NDManager manager = Engine.getInstance().newBaseManager();

        /**
         * Sets the {@link TextData.Configuration} to use for the source text data.
         *
         * @param sourceConfiguration the {@link TextData.Configuration}
         * @return this builder
         */
        public T setSourceConfiguration(Configuration sourceConfiguration) {
            this.sourceConfiguration = sourceConfiguration;
            return self();
        }

        /**
         * Sets the {@link TextData.Configuration} to use for the target text data.
         *
         * @param targetConfiguration the {@link TextData.Configuration}
         * @return this builder
         */
        public T setTargetConfiguration(Configuration targetConfiguration) {
            this.targetConfiguration = targetConfiguration;
            return self();
        }

        /**
         * Sets the optional manager for the dataset (default follows engine default).
         *
         * @param manager the manager
         * @return this builder
         */
        public T optManager(NDManager manager) {
            this.manager = manager.newSubManager();
            return self();
        }
    }
}
