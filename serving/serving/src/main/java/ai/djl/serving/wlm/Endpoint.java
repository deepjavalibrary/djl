/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.wlm;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/** A class that represents a webservice endpoint. */
public class Endpoint {

    private List<ModelInfo> models;
    private Map<String, Integer> map;
    private AtomicInteger position;

    /** Constructs an {@code Endpoint} instance. */
    public Endpoint() {
        models = new ArrayList<>();
        map = new ConcurrentHashMap<>();
        position = new AtomicInteger(0);
    }

    /**
     * Adds a model to the entpoint.
     *
     * @param modelInfo the model to be added
     * @return true if add success
     */
    public synchronized boolean add(ModelInfo modelInfo) {
        String version = modelInfo.getVersion();
        if (version == null) {
            if (models.isEmpty()) {
                map.put("default", 0);
                return models.add(modelInfo);
            }
            return false;
        }
        if (map.containsKey(version)) {
            return false;
        }

        map.put(version, models.size());
        return models.add(modelInfo);
    }

    /**
     * Returns the models associated with the endpoint.
     *
     * @return the models associated with the endpoint
     */
    public List<ModelInfo> getModels() {
        return models;
    }

    /**
     * Removes a model version from the {@code Endpoint}.
     *
     * @param version the model version
     * @return null if the specified version doesn't exist
     */
    public synchronized ModelInfo remove(String version) {
        if (version == null) {
            if (models.isEmpty()) {
                return null;
            }
            ModelInfo model = models.remove(0);
            reIndex();
            return model;
        }
        Integer index = map.remove(version);
        if (index == null) {
            return null;
        }
        ModelInfo model = models.remove((int) index);
        reIndex();
        return model;
    }

    /**
     * Returns the {@code ModelInfo} for the specified version.
     *
     * @param version the version of the model to retrieve
     * @return the {@code ModelInfo} for the specified version
     */
    public ModelInfo get(String version) {
        Integer index = map.get(version);
        if (index == null) {
            return null;
        }
        return models.get(index);
    }

    /**
     * Returns the next version of model to serve the inference request.
     *
     * @return the next version of model to serve the inference request
     */
    public ModelInfo next() {
        int size = models.size();
        if (size == 1) {
            return models.get(0);
        }
        int index = position.getAndUpdate(operand -> (operand + 1) % size);
        return models.get(index);
    }

    private void reIndex() {
        map.clear();
        int size = models.size();
        for (int i = 0; i < size; ++i) {
            ModelInfo modelInfo = models.get(i);
            String version = modelInfo.getVersion();
            if (version != null) {
                map.put(version, i);
            }
        }
    }
}
