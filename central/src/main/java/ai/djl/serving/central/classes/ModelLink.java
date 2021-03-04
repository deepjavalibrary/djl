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
package ai.djl.serving.central.classes;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A class to find the URL link when given a model name.
 *
 * @author anfee1@morgan.edu
 */
public final class ModelLink {

    private static URI base = URI.create("https://mlrepo.djl.ai/");
    private static Map<String, URI> links = new ConcurrentHashMap<>();
    private static final Logger logger = LoggerFactory.getLogger(ModelLink.class);

    private ModelLink() {}

    /**
     * Takes in a model name and returns a Map of download links.
     *
     * @param modelName the connection context
     * @return This returns a map of download links
     * @throws IOException throws an exception
     * @throws ModelNotFoundException throws an exception
     */
    public static Map<String, URI> linkFinder(String modelName)
            throws IOException, ModelNotFoundException {
        Map<Application, List<Artifact>> models = ModelZoo.listModels();
        models.forEach(
                (app, list) -> {
                    list.forEach(
                            artifact -> {
                                if (artifact.getName().equals(modelName)) {
                                    for (Map.Entry<String, Artifact.Item> entry :
                                            artifact.getFiles().entrySet()) {
                                        URI fileUri = URI.create(entry.getValue().getUri());
                                        URI baseUri = artifact.getMetadata().getRepositoryUri();
                                        if (!fileUri.isAbsolute()) {
                                            fileUri = base.resolve(baseUri).resolve(fileUri);
                                        }
                                        try {
                                            links.put(entry.getKey(), fileUri);
                                        } catch (Exception e) {
                                            logger.info(String.valueOf(e));
                                        }
                                    }
                                }
                            });
                });
        return links;
    }
}
