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
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.util.Progress;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

/**
 * Airfoil Self-Noise Data Set from https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise.
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

    private boolean normalize;
    private Map<String, Float> mean;
    private Map<String, Float> std;

    /**
     * Creates an instance of {@code RandomAccessDataset} with the arguments in {@link Builder}.
     *
     * @param builder a builder with the required arguments
     */
    AirfoilRandomAccess(Builder builder) {
        super(builder);
        usage = builder.usage;
        mrl = builder.getMrl();
        normalize = builder.normalize;
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

        if (normalize) {
            mean = new HashMap<>();
            std = new HashMap<>();

            for (Feature feature : features) {
                calculateMean(feature.getName());
                calculateStd(feature.getName());
            }
            for (Feature feature : labels) {
                calculateMean(feature.getName());
                calculateStd(feature.getName());
            }
        }
        prepared = true;
    }

    /** {@inheritDoc} */
    @Override
    public List<String> getColumnNames() {
        return Arrays.asList(COLUMNS).subList(0, 5);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList toNDList(NDManager manager, CSVRecord record, List<Feature> selected) {
        int length = selected.size();
        ByteBuffer bb = manager.allocateDirect(length * 4);
        FloatBuffer buf = bb.asFloatBuffer();
        int index = 0;
        for (Feature feature : selected) {
            String name = feature.getName();
            float value = Float.parseFloat(record.get(name));
            if (normalize) {
                value = (value - mean.get(name)) / std.get(name);
            }
            buf.put(value);
            ++index;
        }
        buf.rewind();
        return new NDList(manager.create(buf, new Shape(length)));
    }

    private void calculateMean(String column) {
        float sum = 0;
        long size = size();
        for (int i = 0; i < size; ++i) {
            CSVRecord record = csvRecords.get(i);
            sum += Float.parseFloat(record.get(column));
        }
        mean.put(column, sum / size);
    }

    private void calculateStd(String column) {
        float average = mean.get(column);
        float sum = 0;
        long size = size();
        for (int i = 0; i < size; ++i) {
            CSVRecord record = csvRecords.get(i);
            sum += (float) Math.pow(Float.parseFloat(record.get(column)) - average, 2);
        }
        std.put(column, (float) Math.sqrt(sum / size));
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
            csvFormat = CSVFormat.TDF.withHeader(COLUMNS).withIgnoreHeaderCase().withTrim();
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
                    addFeature(COLUMNS[i]);
                }
            }
            if (labels.isEmpty()) {
                addNumericLabel("ssoundpres");
            }
            return new AirfoilRandomAccess(this);
        }

        MRL getMrl() {
            return repository.dataset(Tabular.ANY, groupId, artifactId, VERSION);
        }
    }
}
