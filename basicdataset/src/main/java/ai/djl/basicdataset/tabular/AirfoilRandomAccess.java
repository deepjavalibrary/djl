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

import ai.djl.Application.Tabular;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * Airfoil Self-Noise Data Set from <a
 * href="https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise">https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise</a>.
 *
 * <p>1503 instances 6 attributes
 */
public final class AirfoilRandomAccess extends CsvDataset {

    private static final String ARTIFACT_ID = "airfoil";
    private static final String VERSION = "1.0";

    private static final String[] COLUMNS = {
        "freq", "aoa", "chordlen", "freestreamvel", "ssdt", "ssoundpres"
    };

    private MRL mrl;
    private Usage usage;
    private boolean prepared;

    /**
     * Creates an instance of {@code RandomAccessDataset} with the arguments in {@link Builder}.
     *
     * @param builder a builder with the required arguments
     */
    AirfoilRandomAccess(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = mrl.getDefaultArtifact();
        mrl.prepare(artifact);

        Path root = mrl.getRepository().getResourceDirectory(artifact);
        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("airfoil_self_noise.dat");
                break;
            case TEST:
                throw new UnsupportedOperationException("Test data not available.");
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }

        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> getColumnNames() {
        return Arrays.asList(COLUMNS).subList(0, 5);
    }

    /**
     * Creates a builder to build a {@link AirfoilRandomAccess}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@link AirfoilRandomAccess}. */
    public static final class Builder extends CsvBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;
        boolean normalize;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            csvFormat =
                    CSVFormat.TDF
                            .builder()
                            .setHeader(COLUMNS)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .get();
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
         * Sets if normalize the dataset.
         *
         * @param normalize true to normalize the dataset
         * @return the builder
         */
        public Builder optNormalize(boolean normalize) {
            this.normalize = normalize;
            return this;
        }

        /**
         * Returns the available features of this dataset.
         *
         * @return a list of feature names
         */
        public List<String> getAvailableFeatures() {
            return Arrays.asList(COLUMNS);
        }

        /**
         * Adds a feature to the features set.
         *
         * @param name the name of the feature
         * @return this builder
         */
        public Builder addFeature(String name) {
            return addFeature(new Feature(name, true));
        }

        /** {@inheritDoc} */
        @Override
        public AirfoilRandomAccess build() {
            if (features.isEmpty()) {
                for (int i = 0; i < 5; ++i) {
                    addNumericFeature(COLUMNS[i], normalize);
                }
            }
            if (labels.isEmpty()) {
                addNumericLabel("ssoundpres", normalize);
            }
            return new AirfoilRandomAccess(this);
        }

        MRL getMrl() {
            return repository.dataset(Tabular.ANY, groupId, artifactId, VERSION);
        }
    }
}
