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
package ai.djl.basicdataset.nlp;

import ai.djl.Application.NLP;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.CsvDataset;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.csv.CSVFormat;

/**
 * The {@link AmazonReview} dataset contains a {@link ai.djl.Application.NLP#SENTIMENT_ANALYSIS} set
 * of reviews and their sentiment ratings.
 */
public class AmazonReview extends CsvDataset {

    private static final String VERSION = "1.0";
    private static final String ARTIFACT_ID = "amazon_reviews";

    private MRL mrl;
    private String datasetName;
    private boolean prepared;

    /**
     * Creates a new instance of {@link AmazonReview} with the given necessary configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected AmazonReview(Builder builder) {
        super(builder);
        mrl = builder.getMrl();
        datasetName = builder.datasetName;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Map<String, String> filter = new ConcurrentHashMap<>();
        filter.put("dataset", datasetName);
        Artifact artifact = mrl.match(filter);
        mrl.prepare(artifact, progress);

        Path dir = mrl.getRepository().getResourceDirectory(artifact);
        Path csvFile = dir.resolve(artifact.getFiles().values().iterator().next().getName());
        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /**
     * Creates a new builder to build a {@code AmazonReview}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@code AmazonReview}. */
    public static final class Builder extends CsvBuilder<AmazonReview.Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        String datasetName;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            csvFormat = CSVFormat.TDF.withQuote(null).withHeader();
            datasetName = "us_Digital_Software";
        }

        /** {@inheritDoc} */
        @Override
        public Builder self() {
            return this;
        }

        /**
         * Sets the optional repository.
         *
         * @param repository the repository
         * @return this builder
         */
        public Builder optRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Sets optional groupId.
         *
         * @param groupId the groupId}
         * @return this builder
         */
        public Builder optGroupId(String groupId) {
            this.groupId = groupId;
            return this;
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
            return this;
        }

        /**
         * Sets the name of the subset of Amazon Reviews.
         *
         * @param datasetName the name of the dataset
         * @return this builder
         */
        public Builder optDatasetName(String datasetName) {
            this.datasetName = datasetName;
            return this;
        }

        /** {@inheritDoc} */
        @Override
        public AmazonReview build() {
            if (features.isEmpty()) {
                throw new IllegalStateException("Missing features.");
            }
            if (labels.isEmpty()) {
                addNumericLabel("star_rating");
            }
            return new AmazonReview(this);
        }

        MRL getMrl() {
            return repository.dataset(NLP.ANY, groupId, artifactId, VERSION);
        }
    }
}
