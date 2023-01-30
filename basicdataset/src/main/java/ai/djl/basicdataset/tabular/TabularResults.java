/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.List;

/** A list of results from running a tabular model. */
public class TabularResults {

    private List<TabularResult> results;

    /**
     * Constructs a {@link TabularResults} with the given results.
     *
     * @param results the results
     */
    public TabularResults(List<TabularResult> results) {
        this.results = results;
    }

    /**
     * Returns the result for the given feature index.
     *
     * @param index the feature/label index
     * @return the result
     */
    public TabularResult getFeature(int index) {
        return results.get(index);
    }

    /**
     * Returns the result for the given feature name.
     *
     * @param name the feature/label name
     * @return the result
     */
    public TabularResult getFeature(String name) {
        for (TabularResult result : results) {
            if (result.getName().equals(name)) {
                return result;
            }
        }
        throw new IllegalArgumentException(
                "The TabularResults does not contain a result with name " + name);
    }

    /**
     * Returns all of the {@link TabularResult}.
     *
     * @return all of the {@link TabularResult}
     */
    public List<TabularResult> getAll() {
        return results;
    }

    /**
     * Returns the number of results.
     *
     * @return the number of results
     */
    public int size() {
        return results.size();
    }

    /** A single result corresponding to a single feature. */
    public static final class TabularResult {

        private String name;
        private Object result;

        /**
         * Constructs the result.
         *
         * @param name the feature name
         * @param result the computed feature result
         */
        public TabularResult(String name, Object result) {
            this.name = name;
            this.result = result;
        }

        /**
         * Returns the result (feature) name.
         *
         * @return the result (feature) name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the computed result.
         *
         * @return the computed result
         */
        public Object getResult() {
            return result;
        }
    }
}
