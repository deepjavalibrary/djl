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

import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.Featurizers;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.DefaultModelZoo;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * M5 Forecasting - Accuracy from <a
 * href="https://www.kaggle.com/competitions/m5-forecasting-accuracy">https://www.kaggle.com/competitions/m5-forecasting-accuracy</a>
 *
 * <p>To improve the model performance, we coarse graining target of the dataset by summing the sale
 * amount every seven days. And set the column names of sum as 'w_i'. This can reduce occurrence of
 * invalid values 0 and reduce the noise learned by model.
 */
public class M5Forecast extends CsvTimeSeriesDataset {

    private Usage usage;
    private MRL mrl;
    private boolean prepared;
    private Path root;
    private List<Integer> cardinality;

    /**
     * Creates a new instance of {@link M5Forecast} with the given necessary configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    protected M5Forecast(Builder builder) {
        super(builder);
        usage = builder.usage;
        String path = builder.repository.getBaseUri().toString();
        mrl = MRL.undefined(builder.repository, DefaultModelZoo.GROUP_ID, path);
        root = Paths.get(mrl.getRepository().getBaseUri());
        cardinality = builder.cardinality;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        mrl.prepare(null, progress);
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
        String usagePath;
        switch (usage) {
            case TRAIN:
                usagePath = "weekly_sales_train_validation.csv";
                return usagePath;
            case TEST:
                usagePath = "weekly_sales_train_evaluation.csv";
                return usagePath;
            case VALIDATION:
            default:
                throw new UnsupportedOperationException("Data not available.");
        }
    }

    /** Used to build a {@code M5Forecast}. */
    public static class Builder extends CsvBuilder<Builder> {

        Repository repository;
        Usage usage = Usage.TRAIN;
        M5Features mf;
        List<Integer> cardinality;

        Builder() {
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

        /** {@inheritDoc} */
        @Override
        protected Builder self() {
            return this;
        }

        /**
         * Set the repository containing the path.
         *
         * @param repository the repository containing the path
         * @return this builder
         */
        public Builder setRepository(Repository repository) {
            this.repository = repository;
            return this;
        }

        /**
         * Set the optional usage.
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
                                Objects.requireNonNull(
                                        new URL(
                                                        "https://mlrepo.djl.ai/dataset/timeseries/ai/djl/basicdataset/m5forecast-parser/0.1/m5forecast_parser.json")
                                                .openStream());
                        Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8)) {
                    mf = JsonUtils.GSON.fromJson(reader, M5Features.class);
                } catch (IOException e) {
                    throw new AssertionError("Failed to read m5forecast.json from classpath", e);
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
