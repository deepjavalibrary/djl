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

import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.VocabWordEmbedding;
import ai.djl.modality.nlp.embedding.WordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SentenceLengthNormalizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * {@code TextDataset} is an abstract dataset that can be used for datasets for natural language
 * processing where either the source or target are text-based data.
 *
 * <p>The {@code TextDataset} fetches the data in the form of {@link String}, processes the data as
 * required, and creates embeddings for the tokens. Embeddings can be either pre-trained or trained
 * on the go. Pre-trained {@link WordEmbedding} must be set in the {@link Builder}. If no embeddings
 * are set, the dataset creates {@link VocabWordEmbedding} from the {@link Vocabulary} created
 * within the dataset.
 */
public abstract class TextDataset extends RandomAccessDataset {
    TextDatasetParameters textDatasetParameters = new TextDatasetParameters();
    private int embeddingSize;
    private boolean includeValidLength;
    private Tokenizer tokenizer;
    protected long size;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public TextDataset(Builder<?> builder) {
        super(builder);
        this.textDatasetParameters.sourceWordEmbedding = builder.sourceWordEmbedding;
        this.textDatasetParameters.targetWordEmbedding = builder.targetWordEmbedding;
        this.textDatasetParameters.sourceTextProcessors = builder.sourceTextProcessors;
        this.textDatasetParameters.targetTextProcessors = builder.targetTextProcessors;
        this.textDatasetParameters.trainSourceEmbeddings = builder.trainSourceEmbedding;
        this.textDatasetParameters.trainTargetEmbeddings = builder.trainTargetEmbedding;
        this.includeValidLength = builder.includeValidLength;
        this.embeddingSize = builder.embeddingSize;
        this.tokenizer = builder.tokenizer;
    }

    protected NDList embedText(long index, NDManager manager, boolean source)
            throws EmbeddingException {
        NDList data = new NDList();
        NDList dataLengths = new NDList();

        List<String> sentenceTokens = textDatasetParameters.getTextData(source).get(index);
        Integer validLength = textDatasetParameters.getValidLengths(source).get(index);
        for (String token : sentenceTokens) {
            if (textDatasetParameters.trainEmbeddings(source)) {
                data.add(
                        textDatasetParameters
                                .getWordEmbedding(source)
                                .preprocessWordToEmbed(manager, token));
            } else {
                data.add(textDatasetParameters.getWordEmbedding(source).embedWord(manager, token));
            }
            dataLengths.add(manager.create(validLength));
        }
        if (includeValidLength) {
            return new NDList(NDArrays.stack(data), NDArrays.stack(dataLengths));
        }
        return new NDList(NDArrays.stack(data));
    }

    /**
     * Performs pre-processing steps on text data such as tokenising, applying {@link
     * TextProcessor}s, creating vocabulary, and word embeddings.
     *
     * @param textData list of all unprocessed sentences in the dataset
     * @param source whether the text data provided is source or target
     */
    protected void preprocess(List<String> textData, boolean source) {
        Vocabulary.VocabularyBuilder vocabularyBuilder = new Vocabulary.VocabularyBuilder();
        vocabularyBuilder.optMinFrequency(3);
        vocabularyBuilder.optReservedTokens(Arrays.asList("<pad>", "<bos>", "<eos>"));

        size = textData.size();
        Map<Long, List<String>> tokenizedTextData = textDatasetParameters.getTextData(source);
        long index = 0;
        for (String textDatum : textData) {
            List<String> tokens = tokenizer.tokenize(textDatum);
            for (TextProcessor processor : textDatasetParameters.getTextProcessors(source)) {
                tokens = processor.preprocess(tokens);
                if (processor instanceof SentenceLengthNormalizer) {
                    textDatasetParameters.addToValidLengths(
                            index,
                            ((SentenceLengthNormalizer) processor).getLastValidLength(),
                            source);
                }
            }
            vocabularyBuilder.add(tokens);
            tokenizedTextData.put(index, tokens);
            index++;
        }
        Vocabulary vocabulary = vocabularyBuilder.build();
        index = 0;
        while (index < tokenizedTextData.size()) {
            List<String> tokenizedTextDatum = tokenizedTextData.get(index);
            for (int j = 0; j < tokenizedTextDatum.size(); j++) {
                if (!vocabulary.isKnownToken(tokenizedTextDatum.get(j))) {
                    tokenizedTextDatum.set(j, vocabulary.getUnknownToken());
                }
            }
            tokenizedTextData.put(index, tokenizedTextDatum);
            index++;
        }
        textDatasetParameters.setVocabulary(vocabulary, source);
        if (textDatasetParameters.getWordEmbedding(source) == null) {
            textDatasetParameters.setWordEmbedding(
                    new VocabWordEmbedding(vocabulary.newEmbedding(embeddingSize)), source);
            textDatasetParameters.setTrainEmbeddings(true, source);
        }
    }

    /** Abstract Builder that helps build a {@link TextDataset}. */
    public abstract static class Builder<T extends Builder<T>> extends BaseBuilder<T> {
        protected WordEmbedding sourceWordEmbedding;
        protected WordEmbedding targetWordEmbedding;
        protected boolean trainSourceEmbedding;
        protected boolean trainTargetEmbedding;
        protected boolean includeValidLength;
        protected Tokenizer tokenizer;
        protected int embeddingSize;
        protected List<TextProcessor> sourceTextProcessors =
                Arrays.asList(
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(10, false));
        protected List<TextProcessor> targetTextProcessors =
                Arrays.asList(
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(12, true));

        /**
         * Sets the required implementation of {@link WordEmbedding} to get the embeddings for the
         * source.
         *
         * @param wordEmbedding the implementation of {@link WordEmbedding} to source the embeddings
         *     from
         * @param trainSourceEmbedding whether the embeddings need further training
         * @return this builder
         */
        public T optSourceWordEmbedding(WordEmbedding wordEmbedding, boolean trainSourceEmbedding) {
            this.sourceWordEmbedding = wordEmbedding;
            this.trainSourceEmbedding = trainSourceEmbedding;
            return self();
        }

        /**
         * Sets the required implementation of {@link WordEmbedding} to get the embeddings for the
         * target.
         *
         * @param wordEmbedding the implementation of {@link WordEmbedding} to source the embeddings
         *     from
         * @param trainSourceEmbedding whether the embeddings need further training
         * @return this builder
         */
        public T optTargetWordEmbedding(WordEmbedding wordEmbedding, boolean trainSourceEmbedding) {
            this.targetWordEmbedding = wordEmbedding;
            this.trainTargetEmbedding = trainSourceEmbedding;
            return self();
        }

        /**
         * Sets the size of the embeddings. This value must be set if pre-trained {@link
         * WordEmbedding} are not set
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
         * Sets a {@link Tokenizer} to tokenize the input sentences.
         *
         * @param tokenizer the {@link Tokenizer} to be set
         * @return this builder
         */
        public T setTokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
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

    /** Data class that contains parameters of the {@link TextDataset}. */
    private static class TextDatasetParameters {
        Map<Long, List<String>> sourceTextData = new ConcurrentHashMap<>();
        Map<Long, List<String>> targetTextData = new ConcurrentHashMap<>();
        Map<Long, Integer> sourceValidLengths = new ConcurrentHashMap<>();
        Map<Long, Integer> targetValidLengths = new ConcurrentHashMap<>();
        List<TextProcessor> sourceTextProcessors;
        List<TextProcessor> targetTextProcessors;
        WordEmbedding sourceWordEmbedding;
        WordEmbedding targetWordEmbedding;
        Vocabulary sourceVocabulary;
        Vocabulary targetVocabulary;
        boolean trainSourceEmbeddings;
        boolean trainTargetEmbeddings;

        public Map<Long, List<String>> getTextData(boolean source) {
            return source ? sourceTextData : targetTextData;
        }

        public Map<Long, Integer> getValidLengths(boolean source) {
            return source ? sourceValidLengths : targetValidLengths;
        }

        public List<TextProcessor> getTextProcessors(boolean source) {
            return source ? sourceTextProcessors : targetTextProcessors;
        }

        public WordEmbedding getWordEmbedding(boolean source) {
            return source ? sourceWordEmbedding : targetWordEmbedding;
        }

        public Vocabulary getVocabulary(boolean source) {
            return source ? sourceVocabulary : targetVocabulary;
        }

        public boolean trainEmbeddings(boolean source) {
            return source ? trainSourceEmbeddings : trainTargetEmbeddings;
        }

        public void setVocabulary(Vocabulary vocabulary, boolean source) {
            if (source) {
                sourceVocabulary = vocabulary;
            } else {
                targetVocabulary = vocabulary;
            }
        }

        public void setWordEmbedding(WordEmbedding wordEmbedding, boolean source) {
            if (source) {
                sourceWordEmbedding = wordEmbedding;
            } else {
                targetWordEmbedding = wordEmbedding;
            }
        }

        public void addToSentence(Long index, List<String> sentence, boolean source) {
            if (source) {
                sourceTextData.put(index, sentence);
            } else {
                targetTextData.put(index, sentence);
            }
        }

        public void addToValidLengths(Long index, Integer validLength, boolean source) {
            if (source) {
                sourceValidLengths.put(index, validLength);
            } else {
                targetValidLengths.put(index, validLength);
            }
        }

        public void setTrainEmbeddings(boolean trainEmbeddings, boolean source) {
            if (source) {
                trainSourceEmbeddings = trainEmbeddings;
            } else {
                trainTargetEmbeddings = trainEmbeddings;
            }
        }
    }
}
