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
package ai.djl.repository.zoo;

import ai.djl.engine.Engine;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A {@link ModelZoo} that contains models in specified locations. */
public class DefaultModelZoo extends ModelZoo {

    public static final String GROUP_ID = "ai.djl.localmodelzoo";

    private static final Logger logger = LoggerFactory.getLogger(DefaultModelZoo.class);

    private List<ModelLoader> modelLoaders;

    /** Constructs a new {@code LocalModelZoo} instance. */
    public DefaultModelZoo() {}

    /**
     * Constructs a new {@code LocalModelZoo} instance from the given search locations.
     *
     * @param locations a comma separated urls where the models to be loaded from
     */
    public DefaultModelZoo(String locations) {
        modelLoaders = parseLocation(locations);
    }

    /** {@inheritDoc} */
    @Override
    public List<ModelLoader> getModelLoaders() {
        if (modelLoaders != null) {
            return modelLoaders;
        }
        String locations = System.getProperty("ai.djl.repository.zoo.location");
        if (locations != null) {
            return parseLocation(locations);
        }
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Engine.getAllEngines();
    }

    private List<ModelLoader> parseLocation(String locations) {
        String[] urls = locations.split("\\s*,\\s*");
        List<ModelLoader> list = new ArrayList<>(urls.length);
        for (String url : urls) {
            if (!url.isEmpty()) {
                Repository repo = Repository.newInstance(url, url);
                logger.debug("Scanning models in repo: {}, {}", repo.getClass(), url);
                List<MRL> mrls = repo.getResources();
                for (MRL mrl : mrls) {
                    list.add(new BaseModelLoader(mrl));
                }
            } else {
                logger.warn("Model location is empty.");
            }
        }
        return list;
    }
}
