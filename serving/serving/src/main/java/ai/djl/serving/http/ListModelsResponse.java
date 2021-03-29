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
package ai.djl.serving.http;

import java.util.ArrayList;
import java.util.List;

/** A class that holds information about the current registered models. */
public class ListModelsResponse {

    private String nextPageToken;
    private List<ModelItem> models;

    /** Constructs a new {@code ListModelsResponse} instance. */
    public ListModelsResponse() {
        models = new ArrayList<>();
    }

    /**
     * Returns the next page token.
     *
     * @return the next page token
     */
    public String getNextPageToken() {
        return nextPageToken;
    }

    /**
     * Sets the next page token.
     *
     * @param nextPageToken the next page token
     */
    public void setNextPageToken(String nextPageToken) {
        this.nextPageToken = nextPageToken;
    }

    /**
     * Returns a list of models.
     *
     * @return a list of models
     */
    public List<ModelItem> getModels() {
        return models;
    }

    /**
     * Adds the model tp the list.
     *
     * @param modelName the model name
     * @param modelUrl the model url
     */
    public void addModel(String modelName, String modelUrl) {
        models.add(new ModelItem(modelName, modelUrl));
    }

    /** A class that holds model name and url. */
    public static final class ModelItem {

        private String modelName;
        private String modelUrl;

        /** Constructs a new {@code ModelItem} instance. */
        public ModelItem() {}

        /**
         * Constructs a new {@code ModelItem} instance with model name and url.
         *
         * @param modelName the model name
         * @param modelUrl the model url
         */
        public ModelItem(String modelName, String modelUrl) {
            this.modelName = modelName;
            this.modelUrl = modelUrl;
        }

        /**
         * Returns the model name.
         *
         * @return the model name
         */
        public String getModelName() {
            return modelName;
        }

        /**
         * Returns the model url.
         *
         * @return the model url
         */
        public String getModelUrl() {
            return modelUrl;
        }
    }
}
