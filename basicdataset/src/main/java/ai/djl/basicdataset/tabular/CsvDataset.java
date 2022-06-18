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
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.zip.GZIPInputStream;

/** {@code CsvDataset} represents the dataset that stored in a .csv file. */
public class CsvDataset extends TabularDataset {

    protected URL csvUrl;
    protected CSVFormat csvFormat;
    protected List<CSVRecord> csvRecords;

    protected CsvDataset(CsvBuilder<?> builder) {
        super(builder);
        csvUrl = builder.csvUrl;
        csvFormat = builder.csvFormat;
    }

    /** {@inheritDoc} */
    @Override
    protected String getCell(long rowIndex, String featureName) {
        CSVRecord record = csvRecords.get(Math.toIntExact(rowIndex));
        return record.get(featureName);
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
        prepareFeaturizers();
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

    /** Used to build a {@link CsvDataset}. */
    public static class CsvBuilder<T extends CsvBuilder<T>> extends TabularDataset.BaseBuilder<T> {

        protected URL csvUrl;
        protected CSVFormat csvFormat;

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
         * Builds the new {@link CsvDataset}.
         *
         * @return the new {@link CsvDataset}
         */
        public CsvDataset build() {
            validate();
            return new CsvDataset(this);
        }
    }
}
