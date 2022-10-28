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

package ai.djl.timeseries.dataset;

import ai.djl.Application;
import ai.djl.basicdataset.BasicDatasets;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizers;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * M5 Forecasting - Accuracy from <a
 * href="https://www.kaggle.com/competitions/m5-forecasting-accuracy">https://www.kaggle.com/competitions/m5-forecasting-accuracy</a>
 *
 * <p>To improve the model performance, we coarse grain the target of the dataset by summing the
 * sale amount every seven days. And set the column names of sum as 'w_i'. This can reduce
 * occurrence of invalid values 0 and reduce the noise learned by model.
 */
public class M5Forecast extends CsvTimeSeriesDataset {

    private static final String ARTIFACT_ID = "m5forecast";
    private static final String VERSION = "1.0";

    private Usage usage;
    private MRL mrl;
    private boolean prepared;
    private List<Integer> cardinality;

    /**
     * Creates a new instance of {@link M5Forecast} with the given necessary configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected M5Forecast(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
        cardinality = builder.cardinality;
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
        Path csvFile = root.resolve(getUsagePath(usage));

        csvUrl = csvFile.toUri().toURL();
        super.prepare(progress);
        prepared = true;
    }

    /**
     * Return the cardinality of the dataset.
     *
     * @return the cardinality of the dataset
     */
    public List<Integer> getCardinality() {
        return cardinality;
    }

    /**
     * Creates a builder to build a {@link M5Forecast}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    private String getUsagePath(Usage usage) {
        // We coarse graining the data by summing the sale amount every 7 days and rename the .csv
        // file as 'weekly_***'
        switch (usage) {
            case TRAIN:
                return "weekly_sales_train_validation.csv";
            case TEST:
                return "weekly_sales_train_evaluation.csv";
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
    }

    /** Used to build a {@code M5Forecast}. */
    public static class Builder extends CsvBuilder<Builder> {

        Repository repository;
        String groupId;
        String artifactId;
        Usage usage;
        M5Features mf;
        List<Integer> cardinality;

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
            cardinality = new ArrayList<>();
        }

        MRL getMrl() {
            return repository.dataset(Application.Tabular.ANY, groupId, artifactId, VERSION);
        }

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
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
         * Sets the optional usage.
         *
         * @param usage the usage
         * @return this builder
         */
        public Builder optUsage(Usage usage) {
            this.usage = usage;
            return this;
        }

        /**
         * Add a feature to the features set of the filed name.
         *
         * @param name the name of the feature
         * @param fieldName the field name
         * @return this builder
         */
        public Builder addFeature(String name, FieldName fieldName) {
            return addFeature(name, fieldName, false);
        }

        /**
         * Add a feature to the features set of the filed name with onehot encoding.
         *
         * @param name the name of the feature
         * @param fieldName the field name
         * @param onehotEncode true if use onehot encoding
         * @return this builder
         */
        public Builder addFeature(String name, FieldName fieldName, boolean onehotEncode) {
            parseFeatures();
            if (mf.categorical.contains(name)) {
                Map<String, Integer> map = mf.featureToMap.get(name);
                if (map == null) {
                    return addFieldFeature(
                            fieldName,
                            new Feature(name, Featurizers.getStringFeaturizer(onehotEncode)));
                }
                cardinality.add(map.size());
                return addFieldFeature(fieldName, new Feature(name, map, onehotEncode));
            }
            return addFieldFeature(fieldName, new Feature(name, true));
        }

        /**
         * Returns the available features of this dataset.
         *
         * @return a list of feature names
         */
        public List<String> getAvailableFeatures() {
            parseFeatures();
            return mf.featureArray;
        }

        /**
         * Build the new {@code M5Forecast}.
         *
         * @return the new {@code M5Forecast}
         */
        @Override
        public M5Forecast build() {
            validate();
            return new M5Forecast(this);
        }

        private void parseFeatures() {
            if (mf == null) {
                try (InputStream is =
                                M5Forecast.class.getResourceAsStream("m5forecast_parser.json");
                        Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    mf = JsonUtils.GSON.fromJson(reader, M5Features.class);
                } catch (IOException e) {
                    throw new AssertionError(
                            "Failed to read m5forecast_parser.json from classpath", e);
                }
            }
        }
    }

    private static final class M5Features {

        List<String> featureArray;
        Set<String> categorical;
        // categorical = String in featureArray its value indicate a String in featureToMap
        Map<String, Map<String, Integer>> featureToMap;
    }
}
