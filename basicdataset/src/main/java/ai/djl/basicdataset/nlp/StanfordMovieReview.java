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
package ai.djl.basicdataset.nlp;

import ai.djl.Application.NLP;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
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
public class StanfordMovieReview extends TextDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "stanford-movie-review";

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
        this.usage = builder.usage;
        mrl = builder.getMrl();
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
    public void prepare(Progress progress) throws IOException, EmbeddingException {
        if (prepared) {
            return;
        }
        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);
        Path cacheDir = mrl.getRepository().getCacheDirectory();
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

        preprocess(reviewTexts, true);
        prepared = true;
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

        /** Constructs a new builder. */
        public Builder() {
            artifactId = ARTIFACT_ID;
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

        MRL getMrl() {
            return repository.dataset(NLP.ANY, groupId, artifactId, VERSION);
        }
    }
}
