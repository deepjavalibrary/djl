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
package ai.djl.basicdataset.utils;

import java.util.Map;

/** A class contains feature name and its {@code Featurizer}. */
public final class Feature {

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
            featurizer = Featurizers.getNumericFeaturizer();
        } else {
            featurizer = Featurizers.getStringFeaturizer();
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
        this.featurizer = Featurizers.getStringFeaturizer(map, onehotEncode);
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
