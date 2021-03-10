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
package ai.djl.serving.central.utils;

import ai.djl.Application;
import ai.djl.repository.Artifact;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class to find the URIs when given a model name. */
public final class ModelUri {

    // TODO: Use the artifact repository to create base URI
    private static URI base = URI.create("https://mlrepo.djl.ai/");

    private ModelUri() {}

    /**
     * Takes in a model name, artifactId, and groupId to return a Map of download URIs.
     *
     * @param artifactId is the artifactId of the model
     * @param groupId is the groupId of the model
     * @param name is the name of the model
     * @return a map of download URIs
     * @throws IOException if the uri could not be found
     * @throws ModelNotFoundException if Model can not be found
     */
    public static Map<String, URI> uriFinder(String artifactId, String groupId, String name)
            throws IOException, ModelNotFoundException {
        Criteria<?, ?> criteria =
                Criteria.builder()
                        .optModelName(name)
                        .optGroupId(groupId)
                        .optArtifactId(artifactId)
                        .build();
        Map<Application, List<Artifact>> models = ModelZoo.listModels(criteria);
        Map<String, URI> uris = new ConcurrentHashMap<>();
        models.forEach(
                (app, list) -> {
                    list.forEach(
                            artifact -> {
                                for (Map.Entry<String, Artifact.Item> entry :
                                        artifact.getFiles().entrySet()) {
                                    URI fileUri = URI.create(entry.getValue().getUri());
                                    URI baseUri = artifact.getMetadata().getRepositoryUri();
                                    if (!fileUri.isAbsolute()) {
                                        fileUri = base.resolve(baseUri).resolve(fileUri);
                                    }
                                    uris.put(entry.getKey(), fileUri);
                                }
                            });
                });
        return uris;
    }
}
