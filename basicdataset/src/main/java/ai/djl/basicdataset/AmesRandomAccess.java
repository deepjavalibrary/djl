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

import ai.djl.Application.Tabular;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.Resource;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.util.JsonUtils;
import ai.djl.util.Progress;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public class AmesRandomAccess extends RandomAccessDataset {

    private static final Logger logger = LoggerFactory.getLogger(AmesRandomAccess.class);

    private static final String ARTIFACT_ID = "ames";

    private Set<String> enabledFeatures; // enabled features
    private Set<String> categoricalFeatures; // set of categorical features
    private Set<String> disabledFeatures; // disabled features
    private Set<String> oneHotEncode; // set of features that are to be one hot encoded
    // maps from feature to 'category to index' hashmap
    private Map<String, Map<String, Integer>> featureToMap;
    private String label; // only 1 label for now

    private Map<String, FeatureType> featureType;
    private List<CSVRecord> csvRecords; // dataset

    private Usage usage;

    private Resource resource;
    private boolean prepared;

    AmesRandomAccess(Builder builder) {
        super(builder);
        usage = builder.usage;
        MRL mrl = MRL.dataset(Tabular.LINEAR_REGRESSION, builder.groupId, builder.artifactId);
        resource = new Resource(builder.repository, mrl, "1.0");
        label = "saleprice";

        categoricalFeatures = builder.af.categorical;
        enabledFeatures = new HashSet<>(builder.af.featureArray);
        featureToMap = new ConcurrentHashMap<>(builder.af.featureToMap);
        disabledFeatures = new HashSet<>();
        featureType = new ConcurrentHashMap<>();
        oneHotEncode = new HashSet<>();
    }

    /**
     * Creates a builder to build a {@link AmesRandomAccess}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Sets the label(y).
     *
     * @param feature feature to set as the label
     */
    public void setLabel(String feature) {
        feature = feature.toLowerCase();
        if (disabledFeatures.remove(feature)) {
            label = feature;
        }
    }

    /**
     * Get enabled feature set.
     *
     * @return set of string features
     */
    public Set<String> getEnabledFeatures() {
        return enabledFeatures;
    }

    /**
     * Get disabled feature set.
     *
     * @return set of disabled features
     */
    public Set<String> getDisabledFeatures() {
        return disabledFeatures;
    }

    /**
     * Get categorical feature set.
     *
     * @return set of categorical features.
     */
    public Set<String> getCategoricalFeatures() {
        return categoricalFeatures;
    }

    /**
     * Returns the label value for a given index.
     *
     * @param index label index
     * @return label value as a float wrapped in a float array
     */
    public float[] getLabel(int index) {
        // Set for salesprice only currently
        // TODO: Adjust for any feature to be returned
        //        return getValueInt(csvRecords.get(index), label);
        return new float[] {Float.parseFloat(csvRecords.get(index).get(label))};
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
     * @param index index of record
     * @return CSVRecord for given index
     */
    public CSVRecord getCSVRecord(int index) {
        return csvRecords.get(index);
    }

    /**
     * Sets a feature's oneHotEncode status if it is categorical.
     *
     * @param feature feature to be one hot encoded
     * @param enable Enable/Disable oneHotEncoding
     */
    public void setOneHotEncode(String feature, boolean enable) {
        // Check categorical
        if (featureType.get(feature) == FeatureType.CATEGORICAL) {
            if (enable) {
                oneHotEncode.add(feature);
            } else {
                oneHotEncode.remove(feature);
            }
        }
    }

    /**
     * Returns the float value of the record's feature Changes based on categorical or numeric and
     * for categorical if 1 hot encode is enabled.
     *
     * @param record The CSVRecord to get the feature from
     * @param feature The feature value to be acquired
     * @return feature value as a float wrapped in a float array
     */
    public float[] getValueFloat(CSVRecord record, String feature) {
        // if categorical
        if (featureType.get(feature) == FeatureType.NUMERIC) {
            return new float[] {Float.parseFloat(record.get(feature))};
        }

        String value = record.get(feature);
        // convert to numeric
        // check in feature in map
        if (featureToMap.containsKey(feature)) {
            // if 1 hot encode enabled
            Map<String, Integer> categoryTypeToInteger = featureToMap.get(feature);

            if (oneHotEncode.contains(feature)) {
                int categoryTypeCount = categoryTypeToInteger.size();
                float[] oneHotVector = new float[categoryTypeCount];
                // the value of the category can be used as the index for the onehotvector to be
                // true
                oneHotVector[categoryTypeToInteger.get(value)] = 1;
                return oneHotVector;
            }

            // Check value is already seen in feature mapping
            if (categoryTypeToInteger.containsKey(value)) {
                return new float[] {categoryTypeToInteger.get(value)};
            }

            categoryTypeToInteger.put(value, categoryTypeToInteger.size());
            return new float[] {categoryTypeToInteger.size() - 1};
        }

        // Create new map and put in
        Map<String, Integer> map = new ConcurrentHashMap<>();
        featureToMap.put(feature, map);
        map.put(value, 0);
        return new float[] {0};
    }

    /**
     * Returns the size of the feature array(column count).
     *
     * @return feature array size
     */
    public int getFeatureArraySize() {
        int size = enabledFeatures.size(); // get count of enabled features
        // adjust for one hot encoded categories
        for (String feature : oneHotEncode) {
            // Check feature is enabled
            if (enabledFeatures.contains(feature)) {
                size += featureToMap.get(feature).size() - 1;
            }
        }
        return size;
    }

    /**
     * Return the NDArray at index 'index' with the set features.
     *
     * @param manager NDManager to maintain created NDArray
     * @param index Index of wanted NDArray
     * @return NDArray of enabled features
     */
    public NDArray getFeatureNDArray(NDManager manager, int index) {
        CSVRecord record = getCSVRecord(index);
        float[] featureArray = new float[getFeatureArraySize()];

        int i = 0;
        for (String feature : enabledFeatures) {
            float[] values = getValueFloat(record, feature);
            for (float value : values) {
                featureArray[i] = value;
                i++;
            }
        }

        return manager.create(featureArray);
    }

    /** Move all currently set features to available. */
    public void removeAllFeatures() {
        disabledFeatures.addAll(enabledFeatures);
        enabledFeatures.clear();
    }

    /** Move all non set features to be used. */
    public void addAllFeatures() {
        for (String feature : disabledFeatures) {
            addFeature(feature);
        }
    }

    /**
     * Adds a feature of the given type to the used features set.
     *
     * @param feature feature to be added
     * @param type type of feature
     */
    public void addFeature(String feature, FeatureType type) {
        feature = feature.toLowerCase(Locale.getDefault());

        if (disabledFeatures.contains(feature)) {
            featureType.put(feature, type); // add typing
            enabledFeatures.add(feature);
            disabledFeatures.remove(feature);
        } else {
            logger.warn("Unsupported feature: {}", feature);
        }
    }

    /**
     * Adds a feature if it exists.
     *
     * @param feature feature to be enabled
     */
    public void addFeature(String feature) {
        feature = feature.toLowerCase(Locale.getDefault());
        addFeature(feature, getFeatureType(feature));
    }

    /**
     * Sets a feature's type.
     *
     * @param feature feature whose type is to be chagned
     * @param type type for feature to be set to
     */
    public void setFeatureType(String feature, FeatureType type) {
        featureType.put(feature, type);
    }

    /**
     * Removes a feature from the active set if it exists.
     *
     * @param feature feature to be disabled
     */
    public void removeFeature(String feature) {
        feature = feature.toLowerCase();
        if (enabledFeatures.contains(feature)) {
            disabledFeatures.add(feature);
            enabledFeatures.remove(feature);
        }
    }

    /**
     * Returns the feature type.
     *
     * @param feature feature whose type we will get
     * @return type of feature
     */
    public FeatureType getFeatureType(String feature) {
        if (categoricalFeatures.contains(feature)) {
            return FeatureType.CATEGORICAL;
        }
        return FeatureType.NUMERIC;
    }

    /** {@inheritDoc} */
    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Artifact artifact = resource.getDefaultArtifact();
        resource.prepare(artifact, progress);

        Path root =
                resource.getRepository()
                        .getResourceDirectory(artifact)
                        .resolve("house-prices-advanced-regression-techniques");
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

        try (Reader reader = Files.newBufferedReader(csvFile);
                CSVParser csvParser =
                        new CSVParser(
                                reader,
                                CSVFormat.DEFAULT
                                        .withFirstRecordAsHeader()
                                        .withIgnoreHeaderCase()
                                        .withTrim())) {
            csvRecords = csvParser.getRecords();
        }
        prepared = true;
    }

    /** {@inheritDoc} */
    @Override
    protected long availableSize() {
        return csvRecords.size();
    }

    /** A builder to construct a {@link AmesRandomAccess}. */
    public static final class Builder extends BaseBuilder<Builder> {

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
         * Builds the new {@link AmesRandomAccess}.
         *
         * @return the new {@link AmesRandomAccess}
         * @throws IOException for various exceptions depending on the dataset
         */
        public AmesRandomAccess build() throws IOException {
            try (Reader reader =
                    new InputStreamReader(
                            AmesRandomAccess.class.getResourceAsStream("ames.json"),
                            StandardCharsets.UTF_8)) {
                af = JsonUtils.GSON.fromJson(reader, AmesFeatures.class);
            }
            return new AmesRandomAccess(this);
        }
    }

    private static final class AmesFeatures {

        List<String> featureArray;
        Set<String> categorical;
        Map<String, Map<String, Integer>> featureToMap;
    }

    /** An enum represent data type of feature. */
    public enum FeatureType {
        NUMERIC,
        CATEGORICAL
    }
}
