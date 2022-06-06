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

import ai.djl.basicdataset.utils.Feature;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** A abstract class for creating tabular datasets. */
public abstract class TabularDataset extends RandomAccessDataset {

    protected List<Feature> features;
    protected List<Feature> labels;

    /**
     * Creates a new instance of {@link RandomAccessDataset} with the given necessary
     * configurations.
     *
     * @param builder a builder with the necessary configurations
     */
    public TabularDataset(BaseBuilder<?> builder) {
        super(builder);
        features = builder.features;
        labels = builder.labels;
    }

    /** {@inheritDoc} */
    @Override
    public Record get(NDManager manager, long index) {
        NDList data = getRowFeatures(manager, index, features);

        NDList label;
        if (labels.isEmpty()) {
            label = new NDList();
        } else {
            label = getRowFeatures(manager, index, labels);
        }

        return new Record(data, label);
    }

    /**
     * Returns the designated features (either data or label features) from a row.
     *
     * @param manager the manager used to create the arrays
     * @param index the index of the requested data item
     * @param selected the features to pull from the row
     * @return the features formatted as an {@link NDList}
     */
    public abstract NDList getRowFeatures(NDManager manager, long index, List<Feature> selected);

    /**
     * Used to build a {@link TabularDataset}.
     *
     * @param <T> the builder type
     */
    public abstract static class BaseBuilder<T extends BaseBuilder<T>>
            extends RandomAccessDataset.BaseBuilder<T> {

        protected List<Feature> features;
        protected List<Feature> labels;

        protected BaseBuilder() {
            features = new ArrayList<>();
            labels = new ArrayList<>();
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
         * Validates the builder to ensure it is correct.
         *
         * @throws IllegalArgumentException if there is an error with the builder arguments
         */
        protected void validate() {
            if (features.isEmpty()) {
                throw new IllegalArgumentException("Missing features.");
            }
            if (labels.isEmpty()) {
                throw new IllegalArgumentException("Missing labels.");
            }
        }
    }
}
