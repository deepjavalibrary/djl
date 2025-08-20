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
package ai.djl.basicdataset.tabular;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Movielens 100k movie reviews dataset from https://grouplens.org/datasets/movielens/100k/. */
public final class MovieLens100k extends CsvDataset {

    private static final String ARTIFACT_ID = "movielens-100k";
    private static final String VERSION = "1.0";

    private static final String[] USER_FEATURES = {
        "user_id", "user_age", "user_gender", "user_occupation", "user_zipcode"
    };
    private static final String[] MOVIE_FEATURES = {
        "movie_id",
        "movie_title",
        "movie_release_date",
        "movie_video_release_date",
        "imdb_url",
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-fi",
        "thriller",
        "war",
        "western"
    };

    enum HeaderEnum {
        user_id,
        movie_id,
        rating,
        timestamp
    }

    private Usage usage;
    private MRL mrl;
    private boolean prepared;
    private Map<String, Map<String, String>> userFeaturesMap;
    private Map<String, Map<String, String>> movieFeaturesMap;

    MovieLens100k(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
    }

    /** {@inheritDoc} */
    @Override
    public String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        if (HeaderEnum.rating.toString().equals(featureName)) {
            return record.get(HeaderEnum.rating);
        }
        if (featureName.startsWith("user")) {
            String userId = record.get(HeaderEnum.user_id);
            return userFeaturesMap.get(userId).get(featureName);
        }
        String movieId = record.get(HeaderEnum.movie_id);
        return movieFeaturesMap.get(movieId).get(featureName);
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);

        Path dir = mrl.getRepository().getResourceDirectory(artifact);
        Path root = dir.resolve("ml-100k/ml-100k");

        // The actual feature values to use for training/testing are stored in separate files
        Path userFeaturesFile = root.resolve("u.user");
        userFeaturesMap = prepareFeaturesMap(userFeaturesFile, USER_FEATURES);
        Path movieFeaturesFile = root.resolve("u.item");
        movieFeaturesMap = prepareFeaturesMap(movieFeaturesFile, MOVIE_FEATURES);

        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("ua.base");
                break;
            case TEST:
                csvFile = root.resolve("ua.test");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available");
        }

        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    private Map<String, Map<String, String>> prepareFeaturesMap(
            Path featureFile, String[] featureNames) throws IOException {
        URL featureFileUrl = featureFile.toUri().toURL();
        CSVFormat format = CSVFormat.Builder.create(CSVFormat.newFormat('|')).get();
        Reader reader =
                new InputStreamReader(
                        new BufferedInputStream(featureFileUrl.openStream()),
                        StandardCharsets.UTF_8);
        CSVParser csvParser = CSVParser.parse(reader, format);
        List<CSVRecord> featureRecords = csvParser.getRecords();

        Map<String, Map<String, String>> featuresMap = new ConcurrentHashMap<>();
        for (CSVRecord record : featureRecords) {
            Map<String, String> featureValues = new ConcurrentHashMap<>();
            for (int i = 0; i < featureNames.length; i++) {
                featureValues.put(featureNames[i], record.get(i));
            }
            featuresMap.put(record.get(0), featureValues);
        }
        return featuresMap;
    }

    /**
     * Creates a builder to build a {@link MovieLens100k}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@link MovieLens100k}. */
    public static final class Builder extends CsvBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;
        List<String> featureArray =
                new ArrayList<>(
                        Arrays.asList(
                                "user_age",
                                "user_gender",
                                "user_occupation",
                                "user_zipcode",
                                "movie_title",
                                "movie_genres"));

        List<String> movieGenres =
                new ArrayList<>(
                        Arrays.asList(
                                "unknown",
                                "action",
                                "adventure",
                                "animation",
                                "childrens",
                                "comedy",
                                "crime",
                                "documentary",
                                "drama",
                                "fantasy",
                                "film-noir",
                                "horror",
                                "musical",
                                "mystery",
                                "romance",
                                "sci-fi",
                                "thriller",
                                "war",
                                "western"));

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            csvFormat = CSVFormat.TDF.builder().setHeader(HeaderEnum.class).setQuote(null).get();
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }

        /**
         * Sets the optional usage.
         *
         * @param usage the new usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return self();
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return self();
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId}
         * @return this builder
         */
        public Builder optGroupId(String groupId) {
            this.groupId = groupId;
            return self();
        }

        /**
         * Sets the optional artifactId.
         *
         * @param artifactId the artifactId
         * @return this builder
         */
        public Builder optArtifactId(String artifactId) {
            if (artifactId.contains(":")) {
                String[] tokens = artifactId.split(":");
                groupId = tokens[0];
                this.artifactId = tokens[1];
            } else {
                this.artifactId = artifactId;
            }
            return self();
        }

        /**
         * Returns the available features of this dataset.
         *
         * @return a list of feature names
         */
        public List<String> getAvailableFeatures() {
            return featureArray;
        }

        /**
         * Adds a feature to the features set.
         *
         * @param name the name of the feature
         * @return this builder
         */
        public Builder addFeature(String name) {
            if (getAvailableFeatures().contains(name)) {
                switch (name) {
                    case "user_age":
                        addNumericFeature(name);
                        break;
                    case "user_gender":
                    case "user_occupation":
                        addCategoricalFeature(name, true);
                        break;
                    case "user_zipcode":
                    case "movie_title":
                        addCategoricalFeature(name, false);
                        break;
                    case "movie_genres":
                        movieGenres.forEach(this::addNumericFeature);
                        break;
                    default:
                        break;
                }
            } else {
                throw new IllegalArgumentException(
                        String.format(
                                "Provided feature %s is not valid. Valid features are: %s",
                                name, featureArray));
            }
            return self();
        }

        /**
         * Builds the new {@link MovieLens100k}.
         *
         * @return the new {@link MovieLens100k}
         */
        @Override
        public MovieLens100k build() {
            if (features.isEmpty()) {
                featureArray.forEach(this::addFeature);
            }
            if (labels.isEmpty()) {
                addCategoricalLabel("rating", true);
            }
            return new MovieLens100k(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.Tabular.ANY, groupId, artifactId, VERSION);
        }
    }
}
