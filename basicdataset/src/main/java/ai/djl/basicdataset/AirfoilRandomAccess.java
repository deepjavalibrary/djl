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
package ai.djl.basicdataset;

import ai.djl.Application;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.dataset.ZooDataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

/**
 * Airfoil Self-Noise Data Set from https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise.
 *
 * <p>1503 instances 6 attributes
 */
public final class AirfoilRandomAccess extends RandomAccessDataset implements ZooDataset {

    private static final String ARTIFACT_ID = "airfoil";
    private static final String[] FEATURE_ARRAY = {
        "freq", "aoa", "chordlen", "freestreamvel", "ssdt"
    };

    private Set<String> features; // features currently included
    private Set<String> availableFeatures; // features not included
    private String label; // only 1 label for now

    // TODO: add support for more types(in generic)
    // TODO: move away from CSVRecord for storing data
    // as common-csv only always reading(no modifying)
    private List<CSVRecord> csvRecords; // dataset

    private Repository repository;
    private Artifact artifact;
    private Usage usage;
    private boolean prepared;

    private float[][] data;
    private float[] labelArray;

    private Map<String, Integer> stringToIndex;

    /**
     * Creates an instance of {@code RandomAccessDataset} with the arguments in {@link
     * FashionMnist.Builder}.
     *
     * @param builder a builder with the required arguments
     */
    private AirfoilRandomAccess(Builder builder) {
        super(builder);
        repository = builder.repository;
        artifact = builder.artifact;
        usage = builder.usage;

        features = new HashSet<>();
        availableFeatures = new HashSet<>(Arrays.asList(FEATURE_ARRAY));
        label = "ssoundpres";

        stringToIndex = new HashMap<>();
        for (int i = 0; i < FEATURE_ARRAY.length; i++) {
            stringToIndex.put(FEATURE_ARRAY[i], i);
        }
        stringToIndex.put(label, FEATURE_ARRAY.length);
    }

    /** Remove mean and rescale variance to 1 for all features. */
    public void whitenAll() {
        float[] meanArray = new float[FEATURE_ARRAY.length + 1];
        float[] sdArray = new float[FEATURE_ARRAY.length + 1];

        /* Mean Calculation */
        for (CSVRecord record : csvRecords) {
            for (String feature : FEATURE_ARRAY) {
                int index = stringToIndex.get(feature);
                meanArray[index] += getRecordFloat(record, feature);
            }
            int index = stringToIndex.get(label);
            meanArray[index] += getRecordFloat(record, label);
        }

        for (int i = 0; i < meanArray.length; i++) {
            meanArray[i] /= size();
        }
        /* End Mean Calculation */

        /* Standard Deviation Calculation */
        for (CSVRecord record : csvRecords) {
            for (String feature : FEATURE_ARRAY) {
                int index = stringToIndex.get(feature);
                sdArray[index] +=
                        (float) Math.pow(getRecordFloat(record, feature) - meanArray[index], 2);
            }
            int index = stringToIndex.get(label);
            sdArray[index] += (float) Math.pow(getRecordFloat(record, label) - meanArray[index], 2);
        }

        for (int i = 0; i < sdArray.length; i++) {
            sdArray[i] = (float) Math.sqrt(sdArray[i] / csvRecords.size());
        }
        /* End Standard Deviation Calculation */

        data = new float[(int) size()][getFeatureArraySize()];
        labelArray = new float[(int) size()];

        /* Whiten Data */
        for (int i = 0; i < size(); i++) {
            CSVRecord record = csvRecords.get(i);
            for (String feature : FEATURE_ARRAY) {
                int index = stringToIndex.get(feature);
                data[i][index] =
                        (getRecordFloat(record, feature) - meanArray[index]) / sdArray[index];
            }
            labelArray[i] =
                    (getRecordFloat(record, label) - meanArray[FEATURE_ARRAY.length])
                            / sdArray[FEATURE_ARRAY.length];
        }
    }

    /**
     * Gets the feature order of the columns.
     *
     * @return a list of the features in order shown in the FeatureNDArray
     */
    public List<String> getFeatureOrder() {
        return new ArrayList<>(features);
    }

    /**
     * Returns float for a given record and feature.
     *
     * @param record record which holds the raw data
     * @param feature feature to be selected
     * @return float
     */
    public float getRecordFloat(CSVRecord record, String feature) {
        return Float.parseFloat(record.get(feature));
    }

    /**
     * Chooses the 1st N records to be used (not reversible).
     *
     * <p>TODO: make standalone without need for whiten() after to set data[] (speed penalty)
     *
     * @param n number of records to be used starting from the beginning
     */
    public void selectFirstN(int n) {
        csvRecords.subList(n, csvRecords.size()).clear();
    }

    /**
     * Creates a builder to build a {@link AirfoilRandomAccess}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public MRL getMrl() {
        return MRL.dataset(
                Application.Tabular.LINEAR_REGRESSION, BasicDatasets.GROUP_ID, ARTIFACT_ID);
    }

    /** {@inheritDoc} */
    @Override
    public Repository getRepository() {
        return repository;
    }

    /** {@inheritDoc} */
    @Override
    public Artifact getArtifact() {
        return artifact;
    }

    /** {@inheritDoc} */
    @Override
    public Usage getUsage() {
        return usage;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isPrepared() {
        return prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void setPrepared(boolean prepared) {
        this.prepared = prepared;
    }

    /** {@inheritDoc} */
    @Override
    public void useDefaultArtifact() throws IOException {
        artifact = repository.resolve(getMrl(), "1.0", null);
    }

    /**
     * Returns the label value for a given index.
     *
     * @param index index of label
     * @return float value wrapped in a float array
     */
    public float[] getLabel(int index) {
        // Set for sound only currently
        // TODO: Adjust for any feature to be returned
        return new float[] {labelArray[index]};
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        int idx = Math.toIntExact(index);
        NDList d = new NDList(getFeatureNDArray(manager, idx));
        NDList l = new NDList(manager.create(getLabel(idx)));
        return new Record(d, l);
    }

    /**
     * Returns the CSVRecord for a given index.
     *
     * @param index index of desired record
     * @return the requested CSVRecord
     */
    public CSVRecord getCSVRecord(int index) {
        return csvRecords.get(index);
    }

    /**
     * Returns the float value of the record's feature in a float[].
     *
     * @param record The CSVRecord to get the feature from
     * @param feature The feature value to be acquired
     * @return the float value of the feature wrapped in a float array
     */
    public float[] getValueFloat(CSVRecord record, String feature) {
        return new float[] {Float.parseFloat(record.get(feature))};
    }

    /**
     * Returns the size of the feature array(column count).
     *
     * @return size of enabled features
     */
    public int getFeatureArraySize() {
        return features.size(); // get count of enabled features
    }

    /**
     * Return the NDArray at index 'index' with the set features.
     *
     * @param manager NDManager to maintain created NDArray
     * @param index Index of wanted NDArray
     * @return NDArray of features
     */
    public NDArray getFeatureNDArray(NDManager manager, int index) {
        float[] newFeatureArray = new float[getFeatureArraySize()];

        int i = 0;
        for (String feature : features) {
            int featureIndex = stringToIndex.get(feature);
            newFeatureArray[i] = data[index][featureIndex];
            i++;
        }

        return manager.create(newFeatureArray);
    } /* getFeatureNDArray() */

    /** Move all currently set features to available. */
    public void removeAllFeatures() {
        availableFeatures.addAll(features);
        features.clear();
    }

    /** Move all available features to set. */
    public void addAllFeatures() {
        features.addAll(availableFeatures);
        availableFeatures.clear();
    }

    /**
     * Adds a feature if it exists.
     *
     * @param feature requested feature
     */
    public void addFeature(String feature) {
        feature = feature.toLowerCase();
        if (availableFeatures.contains(feature)) {
            availableFeatures.remove(feature);
            features.add(feature);
        }
    }

    /**
     * Removes a feature from the active set if it exists.
     *
     * @param feature to be removed feature
     */
    public void removeFeature(String feature) {
        feature = feature.toLowerCase();
        if (features.contains(feature)) {
            features.remove(feature);
            availableFeatures.add(feature);
        }
    }

    /** {@inheritDoc} */
    @Override
    public void prepareData(Usage usage) throws IOException {
        Path root = repository.getResourceDirectory(artifact);
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

        try (Reader reader = Files.newBufferedReader(csvFile);
                CSVParser csvParser =
                        new CSVParser(
                                reader,
                                CSVFormat.TDF
                                        .withHeader(
                                                "freq",
                                                "aoa",
                                                "chordlen",
                                                "freestreamvel",
                                                "ssdt",
                                                "ssoundpres")
                                        .withIgnoreHeaderCase()
                                        .withTrim())) {
            csvRecords = csvParser.getRecords();
        }

        data = new float[(int) size()][FEATURE_ARRAY.length];
        labelArray = new float[(int) size()];

        // Set data array
        for (int i = 0; i < csvRecords.size(); i++) {
            for (String feature : FEATURE_ARRAY) {
                int featureIndex = stringToIndex.get(feature);
                data[i][featureIndex] = getRecordFloat(getCSVRecord(i), feature);
            }
            labelArray[i] = getRecordFloat(getCSVRecord(i), label);
        }
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return csvRecords.size();
    }

    /** A builder to construct a {@link AirfoilRandomAccess}. */
    public static final class Builder extends BaseBuilder<Builder> {

        Repository repository;
        Artifact artifact;
        Usage usage;

        /** Constructs a new builder. */
        Builder() {
            repository = BasicDatasets.REPOSITORY;
            usage = Usage.TRAIN;
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
         * Sets the optional artifact.
         *
         * @param artifact the artifact
         * @return this builder
         */
        public Builder optArtifact(Artifact artifact) {
            this.artifact = artifact;
            return self();
        }

        /**
         * Builds the new {@link AirfoilRandomAccess}.
         *
         * @return the new {@link AirfoilRandomAccess}
         */
        public AirfoilRandomAccess build() {
            return new AirfoilRandomAccess(this);
        }
    }
}
