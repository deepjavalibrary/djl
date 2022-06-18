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

import ai.djl.Application.Tabular;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizers;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Daily Delhi climate dataset from <a
 * href="https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data">https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data</a>.
 *
 * <p>The Dataset is fully dedicated for the developers who want to train the model on Weather
 * Forecasting for Indian climate. This dataset provides data from 1st January 2013 to 24th April
 * 2017 in the city of Delhi, India. The 4 parameters here are meantemp, humidity, wind_speed,
 * meanpressure.
 */
public class DailyDelhiClimate extends CsvDataset {

    private static final String ARTIFACT_ID = "daily-delhi-climate";
    private static final String VERSION = "3.0";

    private Usage usage;
    private MRL mrl;
    private boolean prepared;

    DailyDelhiClimate(Builder builder) {
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

        Path root = mrl.getRepository().getResourceDirectory(artifact);
        Path csvFile;
        switch (usage) {
            case TRAIN:
                csvFile = root.resolve("DailyDelhiClimateTrain.csv");
                break;
            case TEST:
                csvFile = root.resolve("DailyDelhiClimateTest.csv");
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
     * Creates a builder to build a {@link DailyDelhiClimate}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A builder to construct a {@link DailyDelhiClimate}. */
    public static final class Builder extends CsvBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;
        List<String> featureArray =
                new ArrayList<>(
                        Arrays.asList(
                                "date", "meantemp", "humidity", "wind_speed", "meanpressure"));

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            groupId = BasicDatasets.GROUP_ID;
            artifactId = ARTIFACT_ID;
            usage = Usage.TRAIN;
            csvFormat =
                    CSVFormat.DEFAULT
                            .builder()
                            .setHeader()
                            .setSkipHeaderRecord(true)
                            .setIgnoreHeaderCase(true)
                            .setTrim(true)
                            .build();
        }

        /**
         * Returns this {code Builder} object.
         *
         * @return this {@code BaseBuilder}
         */
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
            if ("date".equals(name)) {
                return addFeature(
                        new Feature(name, Featurizers.getEpochDayFeaturizer("yyyy-MM-dd")));
            } else {
                return addNumericFeature(name);
            }
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
         * Builds the new {@link DailyDelhiClimate}.
         *
         * @return the new {@link DailyDelhiClimate}
         */
        @Override
        public DailyDelhiClimate build() {
            if (features.isEmpty()) {
                for (String name : featureArray) {
                    addFeature(name);
                }
            }
            return new DailyDelhiClimate(this);
        }

        MRL getMrl() {
            return repository.dataset(Tabular.ANY, groupId, artifactId, VERSION);
        }
    }
}
