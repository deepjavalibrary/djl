/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import org.apache.commons.csv.CSVFormat;

/**
 * GoEmotions is a corpus of 58k carefully curated comments extracted from Reddit, with human
 * annotations to 27 emotion categories or Neutral. On top of the raw data, we also include a
 * version filtered based on reter-agreement, which contains a train/test/validation split. The
 * emotion categories are: admiration, amusement, anger, annoyance, approval, caring, confusion,
 * curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear,
 * gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness,
 * surprise.
 */
public class GoEmotions extends CsvDataset {

    private static final String ARTIFACT_ID = "goemotions";
    private static final String VERSION = "1.0";

    private Usage usage;
    private MRL mrl;
    private boolean prepared;

    /**
     * Creates a new instance of {@link GoEmotions}.
     *
     * @param builder the builder object to build from
     */
    GoEmotions(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
    }

    /**
     * Prepares the dataset for use with tracked progress. In this method the TSV file will be
     * parsed. All datasets will be preprocessed.
     *
     * @param progress the progress tracker
     * @throws IOException for various exceptions depending on the dataset
     */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact, progress);

        Path root = mrl.getRepository().getResourceDirectory(artifact);
        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("train.tsv");
                break;
            case TEST:
                csvFile = root.resolve("test.tsv");
                break;
            case VALIDATION:
                csvFile = root.resolve("dev.tsv");
                break;
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /**
     * Creates a builder to build a {@link GoEmotions}.
     *
     * @return a new builder
     */
    public static GoEmotions.Builder builder() {
        return new GoEmotions.Builder();
    }

    /** A builder to construct a {@link GoEmotions}. */
    public static final class Builder extends CsvBuilder<GoEmotions.Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;

        /**
         * The ENUM sets of headers of GoEmotions Datasets.
         */
        public enum HeaderEnum {
            text,
            emotion_id,
            comment_id
        }

        /**
         * Constructs a new builder. adjust the csvFormat to parse TSV file with corresponding
         * headers.
         */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            csvFormat = CSVFormat.TDF.builder().setQuote(null).setHeader(HeaderEnum.class).build();
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
         * Adds a feature to the features set.
         *
         * @param name the name of the feature
         * @return this builder
         */
        public GoEmotions.Builder addFeature(String name) {
            return addFeature(new Feature(name, false));
        }

        /**
         * Returns the available features of this dataset.
         *
         * @return a list of feature names
         */
        public List<String> getAvailableFeatures() {
            return Collections.singletonList("text");
        }

        /**
         * Builds the new {@link GoEmotions}. Add features "text" and label "emotion_id"
         *
         * @return the new {@link GoEmotions}
         */
        @Override
        public GoEmotions build() {
            if (features.isEmpty()) {
                addFeature("text");
            }
            if (labels.isEmpty()) {
                addNumericLabel("emotion_id");
            }
            return new GoEmotions(this);
        }

        MRL getMrl() {
            return repository.dataset(Application.Tabular.ANY, groupId, artifactId, VERSION);
        }
    }
}
