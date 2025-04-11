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

import ai.djl.basicdataset.tabular.utils.DynamicBuffer;
import ai.djl.basicdataset.tabular.utils.Feature;
import ai.djl.basicdataset.tabular.utils.PreparedFeaturizer;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import ai.djl.util.Progress;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

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
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/** {@code CsvTimeSeriesDataset} represents the dataset that store in a .csv file. */
public class CsvTimeSeriesDataset extends TimeSeriesDataset {

    protected PairList<FieldName, List<Feature>> fieldFeatures;
    protected Feature startTimeFeature;
    protected URL csvUrl;
    protected CSVFormat csvFormat;
    protected List<CSVRecord> csvRecords;

    protected CsvTimeSeriesDataset(CsvBuilder<?> builder) {
        super(builder);
        fieldFeatures = builder.fieldFeatures;
        startTimeFeature = builder.startTimeFeatures;
        csvUrl = builder.csvUrl;
        csvFormat = builder.csvFormat;
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
            CSVParser csvParser = CSVParser.parse(reader, csvFormat);
            csvRecords = csvParser.getRecords();
        }
        prepareFeaturizers();
    }

    private InputStream getCsvStream() throws IOException {
        if (csvUrl.getFile().endsWith(".gz")) {
            return new GZIPInputStream(csvUrl.openStream());
        }
        return new BufferedInputStream(csvUrl.openStream());
    }

    /** {@inheritDoc} */
    @Override
    public TimeSeriesData getTimeSeriesData(NDManager manager, long index) {
        TimeSeriesData data = new TimeSeriesData(fieldFeatures.size());
        for (Pair<FieldName, List<Feature>> pair : fieldFeatures) {
            if (!pair.getValue().isEmpty()) {
                data.add(
                        pair.getKey(),
                        getRowFeatures(manager, index, pair.getValue()).singletonOrThrow());
            }
        }

        data.setStartTime(getStartTime(index));
        return data;
    }

    /** Prepares the {@link PreparedFeaturizer}s. */
    protected void prepareFeaturizers() {
        int availableSize = Math.toIntExact(availableSize());
        List<Feature> featuresToPrepare = new ArrayList<>();
        for (List<Feature> list : fieldFeatures.values()) {
            featuresToPrepare.addAll(list);
        }
        for (Feature feature : featuresToPrepare) {
            if (feature.getFeaturizer() instanceof PreparedFeaturizer) {
                PreparedFeaturizer featurizer = (PreparedFeaturizer) feature.getFeaturizer();
                List<String> inputs = new ArrayList<>(Math.toIntExact(availableSize));
                for (int i = 0; i < availableSize; i++) {
                    inputs.add(getCell(i, feature.getName()));
                }
                featurizer.prepare(inputs);
            }
        }
    }

    /**
     * Return the prediction start time for the given index.
     *
     * @param rowIndex the row index
     * @return the start time
     */
    public LocalDateTime getStartTime(long rowIndex) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        TimeFeaturizer featurizer = (TimeFeaturizer) startTimeFeature.getFeaturizer();
        if (featurizer instanceof TimeFeaturizers.ConstantTimeFeaturizer) {
            return featurizer.featurize(null);
        }
        String value = record.get(startTimeFeature.getName());
        return featurizer.featurize(value);
    }

    /**
     * Returns the designated features (either data or label features) from a row.
     *
     * @param manager the manager used to create the arrays
     * @param index the index of the requested data item
     * @param selected the features to pull from the row
     * @return the features formatted as an {@link NDList}
     */
    public NDList getRowFeatures(NDManager manager, long index, List<Feature> selected) {
        DynamicBuffer bb = new DynamicBuffer();
        for (Feature feature : selected) {
            String name = feature.getName();
            String value = getCell(index, name);
            feature.getFeaturizer().featurize(bb, value);
        }
        FloatBuffer buf = bb.getBuffer();
        return new NDList(manager.create(buf, new Shape(bb.getLength())));
    }

    /**
     * Returns a cell in the dataset.
     *
     * @param rowIndex the row index or record index for the cell
     * @param featureName the feature or column of the cell
     * @return the value of the cell at that row and column
     */
    protected String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        return record.get(featureName);
    }

    /** Used to build a {@code CsvTimeSeriesDataset}. */
    public static class CsvBuilder<T extends CsvBuilder<T>> extends TimeSeriesBuilder<T> {

        protected PairList<FieldName, List<Feature>> fieldFeatures;
        protected Feature startTimeFeatures;
        protected URL csvUrl;
        protected CSVFormat csvFormat;

        protected CsvBuilder() {
            fieldFeatures = new PairList<>(DATASET_FIELD_NAMES.length);
            for (FieldName fieldName : DATASET_FIELD_NAMES) {
                fieldFeatures.add(fieldName, new ArrayList<>());
            }
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        protected T self() {
            return (T) this;
        }

        /**
         * Set the optional CSV file path.
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
         * Set the optional CSV file URL.
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
         * Set the CSV file format.
         *
         * @param csvFormat the {@code CSVFormat}
         * @return this builder
         */
        public T setCsvFormat(CSVFormat csvFormat) {
            this.csvFormat = csvFormat;
            return self();
        }

        /**
         * Add the features to the correspongding {@link FieldName}.
         *
         * @param fieldName the correspongding {@link FieldName}
         * @param feature the feature
         * @return this builder
         */
        public T addFieldFeature(FieldName fieldName, Feature feature) {
            if (fieldName == FieldName.START) {
                startTimeFeatures = feature;
            } else if (fieldFeatures.contains(fieldName)) {
                fieldFeatures.get(fieldName).add(feature);
            } else {
                throw new IllegalArgumentException("Unsupported feature field type: " + fieldName);
            }
            return self();
        }

        /**
         * Validate the builder to ensure it is correct.
         *
         * @throws IllegalArgumentException if there is an error with the builder arguments
         */
        protected void validate() {
            if (fieldFeatures.get(FieldName.TARGET).isEmpty()) {
                throw new IllegalArgumentException("Missing target");
            } else if (startTimeFeatures == null) {
                throw new IllegalArgumentException("Missing start time");
            }
        }

        /**
         * Build the new {@link CsvTimeSeriesDataset}.
         *
         * @return the new {@link CsvTimeSeriesDataset}
         */
        public CsvTimeSeriesDataset build() {
            validate();
            return new CsvTimeSeriesDataset(this);
        }
    }
}
