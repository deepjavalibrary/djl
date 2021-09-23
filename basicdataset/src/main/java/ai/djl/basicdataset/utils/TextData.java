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

import ai.djl.basicdataset.nlp.TextDataset;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.AbstractBlock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * {@link TextData} is a utility for managing textual data within a {@link
 * ai.djl.training.dataset.Dataset}.
 *
 * <p>See {@link TextDataset} for an example.
 */
public class TextData {

    private List<NDArray> textEmbeddingList;
    private List<String> rawText;
    private List<TextProcessor> textProcessors;
    private List<String> reservedTokens;
    private TextEmbedding textEmbedding;
    private Vocabulary vocabulary;
    private String unknownToken;
    private int embeddingSize;
    private int size;

    /**
     * Constructs a new {@link TextData}.
     *
     * @param config the configuration for the {@link TextData}
     */
    public TextData(Configuration config) {
        this.textProcessors = config.textProcessors;
        this.textEmbedding = config.textEmbedding;
        this.vocabulary = config.vocabulary;
        this.embeddingSize = config.embeddingSize;
        this.unknownToken = config.unknownToken;
        this.reservedTokens = config.reservedTokens;
    }

    /**
     * Returns a good default {@link Configuration} to use for the constructor with defaults.
     *
     * @return a good default {@link Configuration} to use for the constructor with defaults
     */
    public static Configuration getDefaultConfiguration() {
        List<TextProcessor> defaultTextProcessors =
                Arrays.asList(
                        new SimpleTokenizer(),
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator());

        return new TextData.Configuration()
                .setEmbeddingSize(15)
                .setTextProcessors(defaultTextProcessors)
                .setUnknownToken("<unk>")
                .setReservedTokens(Arrays.asList("<bos>", "<eos>", "<pad>"));
    }

    /**
     * Preprocess the textData into {@link NDArray} by providing the data from the dataset.
     *
     * @param manager the
     * @param newTextData the data from the dataset
     * @throws EmbeddingException if there is an error while embedding input
     */
    public void preprocess(NDManager manager, List<String> newTextData) throws EmbeddingException {
        rawText = newTextData;
        List<List<String>> textData = new ArrayList<>();
        for (String textDatum : newTextData) {
            List<String> tokens = Collections.singletonList(textDatum);
            for (TextProcessor processor : textProcessors) {
                tokens = processor.preprocess(tokens);
            }
            textData.add(tokens);
        }

        if (vocabulary == null) {
            DefaultVocabulary.Builder vocabularyBuilder = DefaultVocabulary.builder();
            vocabularyBuilder
                    .optMinFrequency(3)
                    .optReservedTokens(reservedTokens)
                    .optUnknownToken(unknownToken);
            for (List<String> tokens : textData) {
                vocabularyBuilder.add(tokens);
            }
            vocabulary = vocabularyBuilder.build();
        }

        if (textEmbedding == null) {
            textEmbedding =
                    new TrainableTextEmbedding(
                            new TrainableWordEmbedding(vocabulary, embeddingSize));
        }
        size = textData.size();
        textEmbeddingList = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            List<String> tokenizedTextDatum = textData.get(i);
            for (int j = 0; j < tokenizedTextDatum.size(); j++) {
                tokenizedTextDatum.set(
                        j, vocabulary.getToken(vocabulary.getIndex(tokenizedTextDatum.get(j))));
            }
            textData.set(i, tokenizedTextDatum);
            if (textEmbedding instanceof AbstractBlock) {
                textEmbeddingList.add(
                        manager.create(textEmbedding.preprocessTextToEmbed(tokenizedTextDatum)));
            } else {
                textEmbeddingList.add(textEmbedding.embedText(manager, tokenizedTextDatum));
            }
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
     * Gets the {@link TextEmbedding} used to embed the data with.
     *
     * @return the {@link TextEmbedding}
     */
    public TextEmbedding getTextEmbedding() {
        return textEmbedding;
    }

    /**
     * Sets the embedding size.
     *
     * @param embeddingSize the embedding size
     */
    public void setEmbeddingSize(int embeddingSize) {
        this.embeddingSize = embeddingSize;
    }

    /**
     * Gets the {@link DefaultVocabulary} built while preprocessing the text data.
     *
     * @return the {@link DefaultVocabulary}
     */
    public Vocabulary getVocabulary() {
        if (vocabulary == null) {
            throw new IllegalStateException(
                    "This method must be called after preprocess is called on this object");
        }
        return vocabulary;
    }

    /**
     * Gets the text embedding for the given index of the text input.
     *
     * @param manager the manager for the embedding array
     * @param index the index of the text input
     * @return the {@link NDArray} containing the text embedding
     */
    public NDArray getEmbedding(NDManager manager, long index) {
        NDArray embedding = textEmbeddingList.get(Math.toIntExact(index)).duplicate();
        embedding.attach(manager);
        return embedding;
    }

    /**
     * Gets the raw textual input.
     *
     * @param index the index of the text input
     * @return the raw text
     */
    public String getRawText(long index) {
        return rawText.get(Math.toIntExact(index));
    }

    /**
     * Gets the textual input after preprocessing.
     *
     * @param index the index of the text input
     * @return the list of processed tokens
     */
    public List<String> getProcessedText(long index) {
        List<String> tokens = Collections.singletonList(getRawText(index));
        for (TextProcessor processor : textProcessors) {
            tokens = processor.preprocess(tokens);
        }
        return tokens;
    }

    /**
     * Returns the size of the data.
     *
     * @return the size of the data
     */
    public int getSize() {
        return size;
    }

    /**
     * The configuration for creating a {@link TextData} value in a {@link
     * ai.djl.training.dataset.Dataset}.
     */
    public static final class Configuration {

        private List<TextProcessor> textProcessors;
        private TextEmbedding textEmbedding;
        private Vocabulary vocabulary;
        private Integer embeddingSize;
        private String unknownToken;
        private List<String> reservedTokens;

        /**
         * Sets the {@link TextProcessor}s to use for the text data.
         *
         * @param textProcessors the {@link TextProcessor}s
         * @return this configuration
         */
        public Configuration setTextProcessors(List<TextProcessor> textProcessors) {
            this.textProcessors = textProcessors;
            return this;
        }

        /**
         * Sets the {@link TextEmbedding} to use to embed the text data.
         *
         * @param textEmbedding the {@link TextEmbedding}
         * @return this configuration
         */
        public Configuration setTextEmbedding(TextEmbedding textEmbedding) {
            this.textEmbedding = textEmbedding;
            return this;
        }

        /**
         * Sets the {@link Vocabulary} to use to hold the text data.
         *
         * @param vocabulary the {@link Vocabulary}
         * @return this configuration
         */
        public Configuration setVocabulary(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
            return this;
        }

        /**
         * Sets the size for new {@link TextEmbedding}s.
         *
         * @param embeddingSize the embedding size
         * @return this configuration
         */
        public Configuration setEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        /**
         * Sets the default unknown token.
         *
         * @param unknownToken the {@link String} value of unknown token
         * @return this configuration
         */
        public Configuration setUnknownToken(String unknownToken) {
            this.unknownToken = unknownToken;
            return this;
        }

        /**
         * Sets the list of reserved tokens.
         *
         * @param reservedTokens true to train the text embedding
         * @return this configuration
         */
        public Configuration setReservedTokens(List<String> reservedTokens) {
            this.reservedTokens = reservedTokens;
            return this;
        }

        /**
         * Updates this {@link Configuration} with the non-null values from another configuration.
         *
         * @param other the other configuration to use to update this
         * @return this configuration after updating
         */
        public Configuration update(Configuration other) {
            textProcessors = other.textProcessors != null ? other.textProcessors : textProcessors;
            textEmbedding = other.textEmbedding != null ? other.textEmbedding : textEmbedding;
            vocabulary = other.vocabulary != null ? other.vocabulary : vocabulary;
            embeddingSize = other.embeddingSize != null ? other.embeddingSize : embeddingSize;
            unknownToken = other.unknownToken != null ? other.unknownToken : unknownToken;
            reservedTokens = other.reservedTokens != null ? other.reservedTokens : reservedTokens;
            return this;
        }
    }
}
