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
package ai.djl.basicdataset;

import ai.djl.Application.NLP;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.Record;
import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * The {@link StanfordMovieReview} dataset contains a {@link
 * ai.djl.Application.NLP#SENTIMENT_ANALYSIS} set of movie reviews and their sentiment ratings.
 *
 * <p>The data is sourced from reviews located on IMDB (see <a
 * href="https://ai.stanford.edu/~amaas/data/sentiment/">here</a> for details).
 */
public class StanfordMovieReview extends TextDataset implements ZooDataset {
    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "stanford-movie-review";

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    private List<Boolean> reviewSentiments;
    private List<Integer> reviewImdbScore;

    /**
     * Creates a new instance of {@link StanfordMovieReview} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected StanfordMovieReview(Builder builder) {
        super(builder);
        this.repository = builder.repository;
        this.artifact = builder.artifact;
        this.usage = builder.usage;
    }

    /**
     * Creates a new builder to build a {@link StanfordMovieReview}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return MRL.dataset(NLP.SENTIMENT_ANALYSIS, BasicDatasets.GROUP_ID, ARTIFACT_ID);
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
        Path root = cacheDir.resolve(resourceUri.getPath()).resolve("aclImdb").resolve("aclImdb");

        Path usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = Paths.get("train");
                break;
            case TEST:
                usagePath = Paths.get("test");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }
        usagePath = root.resolve(usagePath);

        List<String> reviewTexts = new ArrayList<>();
        reviewSentiments = new ArrayList<>();
        reviewImdbScore = new ArrayList<>();

        prepareDataSentiment(usagePath.resolve("pos"), true, reviewTexts);
        prepareDataSentiment(usagePath.resolve("neg"), false, reviewTexts);

        try {
            preprocess(reviewTexts, true);
        } catch (EmbeddingException e) {
            throw new IOException(e.getMessage(), e);
        }
    }

    private void prepareDataSentiment(Path path, boolean sentiment, List<String> reviewTexts)
            throws IOException {
        File dir = path.toFile();
        if (!dir.exists()) {
            throw new IllegalArgumentException("Could not find Stanford Movie Review dataset");
        }
        File[] files = dir.listFiles(File::isFile);
        if (files == null) {
            throw new IllegalArgumentException(
                    "Could not find files in Stanford Movie Review dataset");
        }
        for (File reviewFile : files) {
            Path reviewPath = reviewFile.toPath();
            String reviewText = new String(Files.readAllBytes(reviewPath), StandardCharsets.UTF_8);
            String[] splitName = reviewFile.getName().split("\\.")[0].split("_");
            reviewTexts.add(reviewText);
            reviewSentiments.add(sentiment);
            reviewImdbScore.add(Integer.parseInt(splitName[1]));
        }
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        NDList data = new NDList();
        data.add(sourceTextData.getEmbedding(manager, index));
        NDList label =
                new NDList(
                        manager.create(reviewSentiments.get(Math.toIntExact(index)))
                                .toType(DataType.INT32, false));
        return new Record(data, label);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return sourceTextData.getSize();
    }

    /** A builder for a {@link StanfordMovieReview}. */
    public static class Builder extends TextDataset.Builder<Builder> {

        private Repository repository;
        private Artifact artifact;
        private Usage usage;

        /** Constructs a new builder. */
        public Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder setRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public Builder setArtifact(Artifact artifact) {
            this.artifact = artifact;
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder setUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Builds the {@link StanfordMovieReview}.
         *
         * @return the {@link StanfordMovieReview}
         */
        public StanfordMovieReview build() {
            return new StanfordMovieReview(this);
        }
    }
}
