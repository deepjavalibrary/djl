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

import ai.djl.Application;
import ai.djl.modality.nlp.EmbeddingException;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.WordEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SentenceLengthNormalizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.Tokenizer;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import java.io.BufferedReader;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * {@code TatoebaEnglishFrenchDataset} is a English-French machine translation dataset from The
 * Tatoeba Project (http://www.manythings.org/anki/).
 */
public class TatoebaEnglishFrenchDataset extends RandomAccessDataset implements ZooDataset {
    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "tatoeba-en-fr";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    private List<List<String>> sourceSentences;
    private List<Integer> sourceValidLength;
    private List<List<String>> targetSentences;
    private List<Integer> targetValidLength;
    private List<TextProcessor> sourceTextProcessors;
    private List<TextProcessor> targetTextProcessors;
    private WordEmbedding wordEmbedding;
    private boolean trainEmbedding;

    private boolean includeValidLength;
    private Tokenizer tokenizer;

    /**
     * Creates a new instance of {@code TatoebaEnglishFrenchDataset}.
     *
     * @param builder the builder object to build from
     */
    protected TatoebaEnglishFrenchDataset(TatoebaEnglishFrenchDataset.Builder builder) {
        super(builder);
        this.repository = builder.repository;
        this.artifact = builder.artifact;
        this.usage = builder.usage;
        this.wordEmbedding = builder.wordEmbedding;
        this.trainEmbedding = builder.trainEmbedding;
        this.includeValidLength = builder.includeValidLength;
        this.sourceTextProcessors = builder.sourceTextProcessors;
        this.targetTextProcessors = builder.targetTextProcessors;
        this.tokenizer = builder.tokenizer;

        sourceSentences = new ArrayList<>();
        sourceValidLength = new ArrayList<>();
        targetSentences = new ArrayList<>();
        targetValidLength = new ArrayList<>();
    }

    /**
     * Creates a new builder to build a {@link TatoebaEnglishFrenchDataset}.
     *
     * @return a new builder
     */
    public static TatoebaEnglishFrenchDataset.Builder builder() {
        return new TatoebaEnglishFrenchDataset.Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return MRL.dataset(
                Application.NLP.MACHINE_TRANSLATION, BasicDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** {@inheritDoc} */
    @Override
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    @Override
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), VERSION, null);
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) throws IOException {
        Path cacheDir = repository.getCacheDirectory();
        URI resourceUri = artifact.getResourceUri();
        Path root = cacheDir.resolve(resourceUri.getPath());

        Path usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = Paths.get("fra-eng-train.txt");
                break;
            case TEST:
                usagePath = Paths.get("fra-eng-test.txt");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        usagePath = root.resolve(usagePath);

        Vocabulary.VocabularyBuilder sourceVocabularyBuilder = new Vocabulary.VocabularyBuilder();
        sourceVocabularyBuilder.optMinFrequency(3);
        sourceVocabularyBuilder.optReservedTokens(Arrays.asList("<pad>", "<bos>", "<eos>"));

        Vocabulary.VocabularyBuilder targetVocabularyBuilder = new Vocabulary.VocabularyBuilder();
        targetVocabularyBuilder.optMinFrequency(3);
        targetVocabularyBuilder.optReservedTokens(Arrays.asList("<pad>", "<bos>", "<eos>"));

        try (BufferedReader reader = Files.newBufferedReader(usagePath)) {
            String row;
            while ((row = reader.readLine()) != null) {
                String[] sentences = row.split("\t");
                List<String> sourceSentence = tokenizer.tokenize(sentences[0]);
                for (TextProcessor processor : sourceTextProcessors) {
                    sourceSentence = processor.preprocess(sourceSentence);
                    if (processor instanceof SentenceLengthNormalizer) {
                        sourceValidLength.add(
                                ((SentenceLengthNormalizer) processor).getLastValidLength());
                    }
                }

                List<String> targetSentence = tokenizer.tokenize(sentences[1]);
                for (TextProcessor processor : targetTextProcessors) {
                    targetSentence = processor.preprocess(targetSentence);
                    if (processor instanceof SentenceLengthNormalizer) {
                        targetValidLength.add(
                                ((SentenceLengthNormalizer) processor).getLastValidLength());
                    }
                }

                sourceVocabularyBuilder.add(sourceSentence);
                targetVocabularyBuilder.add(targetSentence);
                sourceSentences.add(sourceSentence);
                targetSentences.add(targetSentence);
            }
        }
        Vocabulary sourceVocabulary = sourceVocabularyBuilder.build();
        Vocabulary targetVocabulary = targetVocabularyBuilder.build();

        for (int i = 0; i < sourceSentences.size(); i++) {
            List<String> sentence = sourceSentences.get(i);
            for (int j = 0; j < sentence.size(); j++) {
                if (!sourceVocabulary.isKnownToken(sentence.get(j))) {
                    sentence.set(j, sourceVocabulary.getUnknownToken());
                }
            }
            sourceSentences.set(i, sentence);
        }

        for (int i = 0; i < targetSentences.size(); i++) {
            List<String> sentence = targetSentences.get(i);
            for (int j = 0; j < sentence.size(); j++) {
                if (!targetVocabulary.isKnownToken(sentence.get(j))) {
                    sentence.set(j, targetVocabulary.getUnknownToken());
                }
            }
            targetSentences.set(i, sentence);
        }
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) throws EmbeddingException {
        NDList data = new NDList();
        NDList dataLengths = new NDList();
        NDList target = new NDList();
        NDList targetLengths = new NDList();

        List<String> sourceSentence = sourceSentences.get((int) index);
        List<String> targetSentence = targetSentences.get((int) index);
        for (String token : sourceSentence) {
            if (trainEmbedding) {
                data.add(wordEmbedding.preprocessWordToEmbed(manager, token));
            } else {
                data.add(wordEmbedding.embedWord(manager, token));
            }
            if (includeValidLength) {
                dataLengths.add(manager.create(sourceValidLength.get((int) index)));
            }
        }
        for (String token : targetSentence) {
            if (trainEmbedding) {
                target.add(wordEmbedding.preprocessWordToEmbed(manager, token));
            } else {
                target.add(wordEmbedding.embedWord(manager, token));
            }
            if (includeValidLength) {
                targetLengths.add(manager.create(targetValidLength.get((int) index)));
            }
        }
        if (includeValidLength) {
            return new Record(
                    new NDList(NDArrays.stack(data), NDArrays.stack(dataLengths)),
                    new NDList(NDArrays.stack(target), NDArrays.stack(targetLengths)));
        }
        return new Record(new NDList(NDArrays.stack(data)), new NDList(NDArrays.stack(target)));
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return sourceSentences.size();
    }

    /** A builder for a {@link TatoebaEnglishFrenchDataset}. */
    public static class Builder extends BaseBuilder<Builder> {
        private Repository repository;
        private Artifact artifact;
        private Usage usage;
        protected List<TextProcessor> sourceTextProcessors =
                Arrays.asList(
                        new LowerCaseConvertor(Locale.ENGLISH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(10, false));
        protected List<TextProcessor> targetTextProcessors =
                Arrays.asList(
                        new LowerCaseConvertor(Locale.FRENCH),
                        new PunctuationSeparator(),
                        new SentenceLengthNormalizer(12, true));
        protected WordEmbedding wordEmbedding;
        protected boolean trainEmbedding;
        protected boolean includeValidLength;
        protected Tokenizer tokenizer;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
        }

        /** {@inheritDoc} */
        @Override
        public TatoebaEnglishFrenchDataset.Builder self() {
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public TatoebaEnglishFrenchDataset.Builder optUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public TatoebaEnglishFrenchDataset.Builder optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public TatoebaEnglishFrenchDataset.Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }

        /**
         * Sets the required implementation of {@link WordEmbedding} to source the embeddings from.
         *
         * @param wordEmbedding the implementation of {@link WordEmbedding} to source the embeddings
         *     from
         * @param trainEmbedding whether the embeddings need further training
         * @return this builder
         */
        public Builder setEmbedding(WordEmbedding wordEmbedding, boolean trainEmbedding) {
            this.wordEmbedding = wordEmbedding;
            this.trainEmbedding = trainEmbedding;
            return self();
        }

        /**
         * Sets the required parameter whether to include the valid length as part of data in the
         * {@code Record}.
         *
         * @param includeValidLength whether to include the valid length as part of data
         * @return this builder
         */
        public Builder setValidLength(boolean includeValidLength) {
            this.includeValidLength = includeValidLength;
            return self();
        }

        /**
         * Sets a {@link Tokenizer} to tokenize the input sentences.
         *
         * @param tokenizer the {@link Tokenizer} to be set
         * @return this builder
         */
        public Builder setTokenizer(Tokenizer tokenizer) {
            this.tokenizer = tokenizer;
            return self();
        }

        /**
         * Sets the list of {@link TextProcessor}s to be used on the source language input. The
         * order of {@link TextProcessor} in the list can make a difference.
         *
         * @param sourceTextProcessors the list of {@link TextProcessor}s to be set
         * @return this builder
         */
        public Builder optSourceTextProcessors(List<TextProcessor> sourceTextProcessors) {
            this.sourceTextProcessors = sourceTextProcessors;
            return self();
        }

        /**
         * Sets a {@link TextProcessor} to be used on the source language input. The order in which
         * {@link TextProcessor} is added can make a difference.
         *
         * @param sourceTextProcessor the {@link TextProcessor} to be set
         * @return this builder
         */
        public Builder optSourceTextProcessor(TextProcessor sourceTextProcessor) {
            this.sourceTextProcessors.add(sourceTextProcessor);
            return self();
        }

        /**
         * Sets the list of {@link TextProcessor}s to be used on the target language input. The
         * order of {@link TextProcessor} * in the list can make a difference.
         *
         * @param targetTextProcessors the list of {@link TextProcessor}s to be set
         * @return this builder
         */
        public Builder optTargetTextProcessors(List<TextProcessor> targetTextProcessors) {
            this.targetTextProcessors = targetTextProcessors;
            return self();
        }

        /**
         * Sets a {@link TextProcessor} to be used on the target language input. The order in which
         * {@link TextProcessor} is added can make a difference.
         *
         * @param targetTextProcessor the {@link TextProcessor} to be set
         * @return this builder
         */
        public Builder optTargetTextProcessor(TextProcessor targetTextProcessor) {
            this.targetTextProcessors.add(targetTextProcessor);
            return self();
        }

        /**
         * Builds the {@link TatoebaEnglishFrenchDataset}.
         *
         * @return the {@link TatoebaEnglishFrenchDataset}
         */
        public TatoebaEnglishFrenchDataset build() {
            return new TatoebaEnglishFrenchDataset(this);
        }
    }
}
