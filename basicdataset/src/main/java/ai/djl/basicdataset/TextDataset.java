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
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.SimpleTextEmbedding;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SentenceLengthNormalizer;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * {@code TextDataset} is an abstract dataset that can be used for datasets for natural language
 * processing where either the source or target are text-based data.
 *
 * <p>The {@code TextDataset} fetches the data in the form of {@link String}, processes the data as
 * required, and creates embeddings for the tokens. Embeddings can be either pre-trained or trained
 * on the go. Pre-trained {@link TextEmbedding} must be set in the {@link Builder}. If no embeddings
 * are set, the dataset creates {@link TrainableWordEmbedding} based {@link SimpleTextEmbedding}
 * from the {@link Vocabulary} created within the dataset.
 */
public abstract class TextDataset extends RandomAccessDataset {

    protected TextData sourceTextData;
    protected TextData targetTextData;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public TextDataset(Builder<?> builder) {
        super(builder);

        sourceTextData = new TextData();
        sourceTextData.setTextEmbedding(builder.sourceTextEmbedding);
        sourceTextData.setTrainEmbedding(builder.trainSourceEmbedding);
        sourceTextData.setIncludeValidLength(builder.includeValidLength);
        sourceTextData.setEmbeddingSize(builder.embeddingSize);
        sourceTextData.setTextProcessors(builder.sourceTextProcessors);

        targetTextData = new TextData();
        targetTextData.setTextEmbedding(builder.targetTextEmbedding);
        targetTextData.setTrainEmbedding(builder.trainTargetEmbedding);
        targetTextData.setIncludeValidLength(builder.includeValidLength);
        targetTextData.setEmbeddingSize(builder.embeddingSize);
        targetTextData.setTextProcessors(builder.targetTextProcessors);
    }

    protected NDList embedText(long index, NDManager manager, boolean source)
            throws EmbeddingException {
        TextData textData = source ? sourceTextData : targetTextData;
        return textData.embedText(index, manager);
    }

    /**
     * Performs pre-processing steps on text data such as tokenising, applying {@link
     * TextProcessor}s, creating vocabulary, and word embeddings.
     *
     * @param newTextData list of all unprocessed sentences in the dataset
     * @param source whether the text data provided is source or target
     */
    protected void preprocess(List<String> newTextData, boolean source) {
        TextData textData = source ? sourceTextData : targetTextData;
        textData.preprocess(newTextData);
    }

    /** Abstract Builder that helps build a {@link TextDataset}. */
    public abstract static class Builder<T extends Builder<T>> extends BaseBuilder<T> {

        protected TextEmbedding sourceTextEmbedding;
        protected TextEmbedding targetTextEmbedding;
        protected boolean trainSourceEmbedding;
        protected boolean trainTargetEmbedding;
        protected boolean includeValidLength;
        protected int embeddingSize;
        protected List<TextProcessor> sourceTextProcessors =
                Arrays.asList(
                        new SimpleTokenizer(),
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(10, false));
        protected List<TextProcessor> targetTextProcessors =
                Arrays.asList(
                        new SimpleTokenizer(),
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(12, true));

        /**
         * Sets the required implementation of {@link TextEmbedding} to get the embeddings for the
         * source.
         *
         * @param textEmbedding the implementation of {@link TextEmbedding} to source the embeddings
         *     from
         * @param trainSourceEmbedding whether the embeddings need further training
         * @return this builder
         */
        public T optSourceTextEmbedding(TextEmbedding textEmbedding, boolean trainSourceEmbedding) {
            this.sourceTextEmbedding = textEmbedding;
            this.trainSourceEmbedding = trainSourceEmbedding;
            return self();
        }

        /**
         * Sets the required implementation of {@link TextEmbedding} to get the embeddings for the
         * target.
         *
         * @param textEmbedding the implementation of {@link TextEmbedding} to source the embeddings
         *     from
         * @param trainSourceEmbedding whether the embeddings need further training
         * @return this builder
         */
        public T optTargetTextEmbedding(TextEmbedding textEmbedding, boolean trainSourceEmbedding) {
            this.targetTextEmbedding = textEmbedding;
            this.trainTargetEmbedding = trainSourceEmbedding;
            return self();
        }

        /**
         * Sets the size of the embeddings. This value must be set if pre-trained {@link
         * TextEmbedding} are not set
         *
         * @param embeddingSize the size of the embeddings
         * @return this builder
         */
        public T optEmbeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return self();
        }

        /**
         * Sets the required parameter whether to include the valid length as part of data in the
         * {@code Record}.
         *
         * @param includeValidLength whether to include the valid length as part of data
         * @return this builder
         */
        public T setValidLength(boolean includeValidLength) {
            this.includeValidLength = includeValidLength;
            return self();
        }

        /**
         * Sets the list of {@link TextProcessor}s to be used on the source input. The order of
         * {@link TextProcessor} in the list can make a difference.
         *
         * @param sourceTextProcessors the list of {@link TextProcessor}s to be set
         * @return this builder
         */
        public T optSourceTextProcessors(List<TextProcessor> sourceTextProcessors) {
            this.sourceTextProcessors = sourceTextProcessors;
            return self();
        }

        /**
         * Sets a {@link TextProcessor} to be used on the source input. The order in which {@link
         * TextProcessor} is added can make a difference.
         *
         * @param sourceTextProcessor the {@link TextProcessor} to be set
         * @return this builder
         */
        public T addSourceTextProcessor(TextProcessor sourceTextProcessor) {
            if (sourceTextProcessors == null) {
                sourceTextProcessors = new ArrayList<>();
            }
            sourceTextProcessors.add(sourceTextProcessor);
            return self();
        }

        /**
         * Sets the list of {@link TextProcessor}s to be used on the target. The order of {@link
         * TextProcessor} * in the list can make a difference.
         *
         * @param targetTextProcessors the list of {@link TextProcessor}s to be set
         * @return this builder
         */
        public T optTargetTextProcessors(List<TextProcessor> targetTextProcessors) {
            this.targetTextProcessors = targetTextProcessors;
            return self();
        }

        /**
         * Adds a {@link TextProcessor} to the existing list of {@link TextProcessor} to be used on
         * the target. The order in which {@link TextProcessor} is added can make a difference.
         *
         * @param targetTextProcessor the {@link TextProcessor} to be set
         * @return this builder
         */
        public T addTargetTextProcessor(TextProcessor targetTextProcessor) {
            if (targetTextProcessors == null) {
                targetTextProcessors = new ArrayList<>();
            }
            targetTextProcessors.add(targetTextProcessor);
            return self();
        }
    }
}
