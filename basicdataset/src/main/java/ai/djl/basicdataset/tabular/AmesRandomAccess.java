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
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.csv.CSVFormat;

/**
 * Ames house pricing dataset from
 * https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data.
 *
 * <p>80 features
 *
 * <p>Training Set: 1460 Records
 *
 * <p>Test Set: 1459 Records
 *
 * <p>Can enable/disable features Set one hot vector for categorical variables
 */
public class AmesRandomAccess extends CsvDataset {

    private static final String ARTIFACT_ID = "ames";
    private static final String VERSION = "1.0";

    private Usage usage;
    private MRL mrl;
    private boolean prepared;

    AmesRandomAccess(Builder builder) {
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
        mrl.prepare(artifact, progress);

        Path dir = mrl.getRepository().getResourceDirectory(artifact);
        Path root = dir.resolve("house-prices-advanced-regression-techniques");
        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("train.csv");
                break;
            case TEST:
                csvFile = root.resolve("test.csv");
                break;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Validation data not available.");
        }

        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /**
     * Creates a builder to build a {@link AmesRandomAccess}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@link AmesRandomAccess}. */
    public static final class Builder extends CsvBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;
        AmesFeatures af;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            csvFormat =
                    CSVFormat.DEFAULT.withFirstRecordAsHeader().withIgnoreHeaderCase().withTrim();
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
        public Builder addFeature(String name) {
            return addFeature(name, false);
        }

        /**
         * Adds a feature to the features set with onehot encoding.
         *
         * @param name the name of the feature
         * @param onehotEncode true if use onehot encoding
         * @return this builder
         */
        public Builder addFeature(String name, boolean onehotEncode) {
            parseFeatures();
            if (af.categorical.contains(name)) {
                Map<String, Integer> map = af.featureToMap.get(name);
                if (map == null) {
                    return addCategoricalFeature(name);
                }
                return addCategoricalFeature(name, map, onehotEncode);
            }
            return addNumericFeature(name);
        }

        /**
         * Returns the available features of this dataset.
         *
         * @return a list of feature names
         */
        public List<String> getAvailableFeatures() {
            parseFeatures();
            return af.featureArray;
        }

        /**
         * Builds the new {@link AmesRandomAccess}.
         *
         * @return the new {@link AmesRandomAccess}
         */
        @Override
        public AmesRandomAccess build() {
            if (features.isEmpty()) {
                parseFeatures();
                for (String name : af.featureArray) {
                    addFeature(name);
                }
            }
            if (labels.isEmpty()) {
                addNumericLabel("saleprice");
            }
            return new AmesRandomAccess(this);
        }

        private void parseFeatures() {
            if (af == null) {
                try (InputStream is = AmesRandomAccess.class.getResourceAsStream("ames.json");
                        Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    af = JsonUtils.GSON.fromJson(reader, AmesFeatures.class);
                } catch (IOException e) {
                    throw new AssertionError("Failed to read ames.json from classpath", e);
                }
            }
        }

        MRL getMrl() {
            return repository.dataset(Tabular.ANY, groupId, artifactId, VERSION);
        }
    }

    private static final class AmesFeatures {

        List<String> featureArray;
        Set<String> categorical;
        Map<String, Map<String, Integer>> featureToMap;
    }
}
