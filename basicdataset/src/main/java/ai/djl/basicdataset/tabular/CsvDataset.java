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
package ai.djl.basicdataset.tabular;

import ai.djl.basicdataset.utils.DynamicBuffer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.Progress;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/** {@code CsvDataset} represents the dataset that stored in a .csv file. */
public class CsvDataset extends RandomAccessDataset {

    private static final Featurizer NUMERIC_FEATURIZER = new NumericFeaturizer();

    protected URL csvUrl;
    protected CSVFormat csvFormat;
    protected List<Feature> features;
    protected List<Feature> labels;
    protected List<CSVRecord> csvRecords;

    protected CsvDataset(CsvBuilder<?> builder) {
        super(builder);
        csvUrl = builder.csvUrl;
        csvFormat = builder.csvFormat;
        features = builder.features;
        labels = builder.labels;
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        CSVRecord csvRecord = csvRecords.get(Math.toIntExact(index));
        NDList data = toNDList(manager, csvRecord, features);
        NDList label = toNDList(manager, csvRecord, labels);

        return new Record(data, label);
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return csvRecords.size();
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        try (Reader reader = new InputStreamReader(getCsvStream(), StandardCharsets.UTF_8)) {
            CSVParser csvParser = new CSVParser(reader, csvFormat);
            csvRecords = csvParser.getRecords();
        }
    }

    private InputStream getCsvStream() throws IOException {
        if (csvUrl.getFile().endsWith(".gz")) {
            return new GZIPInputStream(csvUrl.openStream());
        }
        return new BufferedInputStream(csvUrl.openStream());
    }

    /**
     * Creates a builder to build a {@link AmesRandomAccess}.
     *
     * @return a new builder
     */
    public static CsvBuilder<?> builder() {
        return new CsvBuilder<>();
    }

    /**
     * Returns the column names of the CSV file.
     *
     * @return a list of column name
     */
    public List<String> getColumnNames() {
        if (csvRecords.isEmpty()) {
            return Collections.emptyList();
        }
        return csvRecords.get(0).getParser().getHeaderNames();
    }

    protected NDList toNDList(NDManager manager, CSVRecord record, List<Feature> selected) {
        DynamicBuffer bb = new DynamicBuffer();
        for (Feature feature : selected) {
            String name = feature.getName();
            String value = record.get(name);
            feature.featurizer.featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(manager.create(buf, new Shape(bb.getLength())));
    }

    /** Used to build a {@link CsvDataset}. */
    public static class CsvBuilder<T extends CsvBuilder<T>> extends BaseBuilder<T> {

        protected URL csvUrl;
        protected CSVFormat csvFormat;
        protected List<Feature> features;
        protected List<Feature> labels;

        protected CsvBuilder() {
            features = new ArrayList<>();
            labels = new ArrayList<>();
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        protected T self() {
            return (T) this;
        }

        /**
         * Sets the optional CSV file path.
         *
         * @param csvFile the CSV file path
         * @return this builder
         */
        public T optCsvFile(Path csvFile) {
            try {
                this.csvUrl = csvFile.toAbsolutePath().toUri().toURL();
            } catch (MalformedURLException e) {
                throw new IllegalArgumentException("Invalid file path: " + csvFile, e);
            }
            return self();
        }

        /**
         * Sets the optional CSV file URL.
         *
         * @param csvUrl the CSV file URL
         * @return this builder
         */
        public T optCsvUrl(String csvUrl) {
            try {
                this.csvUrl = new URL(csvUrl);
            } catch (MalformedURLException e) {
                throw new IllegalArgumentException("Invalid url: " + csvUrl, e);
            }
            return self();
        }

        /**
         * Sets the CSV file format.
         *
         * @param csvFormat the {@code CSVFormat}
         * @return this builder
         */
        public T setCsvFormat(CSVFormat csvFormat) {
            this.csvFormat = csvFormat;
            return self();
        }

        /**
         * Adds the features to the feature set.
         *
         * @param features the features
         * @return this builder
         */
        public T addFeature(Feature... features) {
            Collections.addAll(this.features, features);
            return self();
        }

        /**
         * Adds a numeric feature to the feature set.
         *
         * @param name the feature name
         * @return this builder
         */
        public T addNumericFeature(String name) {
            features.add(new Feature(name, true));
            return self();
        }

        /**
         * Adds a categorical feature to the feature set.
         *
         * @param name the feature name
         * @return this builder
         */
        public T addCategoricalFeature(String name) {
            features.add(new Feature(name, false));
            return self();
        }

        /**
         * Adds a categorical feature to the feature set with specified mapping.
         *
         * @param name the feature name
         * @param map a map contains categorical value maps to index
         * @param onehotEncode true if use onehot encode
         * @return this builder
         */
        public T addCategoricalFeature(
                String name, Map<String, Integer> map, boolean onehotEncode) {
            features.add(new Feature(name, map, onehotEncode));
            return self();
        }

        /**
         * Adds the features to the label set.
         *
         * @param labels the labels
         * @return this builder
         */
        public T addLabel(Feature... labels) {
            Collections.addAll(this.labels, labels);
            return self();
        }

        /**
         * Adds a number feature to the label set.
         *
         * @param name the label name
         * @return this builder
         */
        public T addNumericLabel(String name) {
            labels.add(new Feature(name, true));
            return self();
        }

        /**
         * Adds a categorical feature to the label set.
         *
         * @param name the feature name
         * @return this builder
         */
        public T addCategoricalLabel(String name) {
            labels.add(new Feature(name, true));
            return self();
        }

        /**
         * Adds a categorical feature to the feature set with specified mapping.
         *
         * @param name the feature name
         * @param map a map contains categorical value maps to index
         * @param onehotEncode true if use onehot encode
         * @return this builder
         */
        public T addCategoricalLabel(String name, Map<String, Integer> map, boolean onehotEncode) {
            labels.add(new Feature(name, map, onehotEncode));
            return self();
        }

        /**
         * Builds the new {@link CsvDataset}.
         *
         * @return the new {@link CsvDataset}
         */
        public CsvDataset build() {
            if (features.isEmpty()) {
                throw new IllegalArgumentException("Missing features.");
            }
            if (labels.isEmpty()) {
                throw new IllegalArgumentException("Missing labels.");
            }
            return new CsvDataset(this);
        }
    }

    /** An interface that convert String to numeric data. */
    public interface Featurizer {

        /**
         * Puts encoded data into the float buffer.
         *
         * @param buf the float buffer to be filled
         * @param input the string input
         */
        void featurize(DynamicBuffer buf, String input);
    }

    /** A class contains feature name and its {@code Featurizer}. */
    public static final class Feature {

        String name;
        Featurizer featurizer;

        /**
         * Constructs a {@code Feature} instance.
         *
         * @param name the feature name
         * @param featurizer the {@code Featurizer}
         */
        public Feature(String name, Featurizer featurizer) {
            this.name = name;
            this.featurizer = featurizer;
        }

        /**
         * Constructs a {@code Feature} instance.
         *
         * @param name the feature name
         * @param numeric true if input is numeric data
         */
        public Feature(String name, boolean numeric) {
            this.name = name;
            if (numeric) {
                featurizer = NUMERIC_FEATURIZER;
            } else {
                featurizer = new StringFeaturizer();
            }
        }

        /**
         * Constructs a {@code Feature} instance.
         *
         * @param name the feature name
         * @param map a map contains categorical value maps to index
         * @param onehotEncode true if use onehot encode
         */
        public Feature(String name, Map<String, Integer> map, boolean onehotEncode) {
            this.name = name;
            this.featurizer = new StringFeaturizer(map, onehotEncode);
        }

        /**
         * Returns the feature name.
         *
         * @return the feature name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the {@code Featurizer}.
         *
         * @return the {@code Featurizer}
         */
        public Featurizer getFeaturizer() {
            return featurizer;
        }
    }

    private static final class NumericFeaturizer implements Featurizer {

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            buf.put(Float.parseFloat(input));
        }
    }

    private static final class StringFeaturizer implements Featurizer {

        private Map<String, Integer> map;
        private boolean onehotEncode;
        private boolean autoMap;

        StringFeaturizer() {
            this.map = new HashMap<>();
            this.autoMap = true;
        }

        StringFeaturizer(Map<String, Integer> map, boolean onehotEncode) {
            this.map = map;
            this.onehotEncode = onehotEncode;
        }

        /** {@inheritDoc} */
        @Override
        public void featurize(DynamicBuffer buf, String input) {
            if (onehotEncode) {
                for (int i = 0; i < map.size(); ++i) {
                    buf.put(i == map.get(input) ? 1 : 0);
                }
                return;
            }

            Integer index = map.get(input);
            if (index != null) {
                buf.put(index);
                return;
            }

            if (!autoMap) {
                throw new IllegalArgumentException("Value: " + input + " not found in the map.");
            }
            int value = map.size();
            map.put(input, value);
            buf.put(value);
        }
    }
}
