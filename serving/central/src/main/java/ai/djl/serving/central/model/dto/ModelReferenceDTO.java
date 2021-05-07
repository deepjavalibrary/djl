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
package ai.djl.serving.central.model.dto;

import ai.djl.repository.Artifact;

/**
 * reference to a model without transferring whole model information.
 *
 * <p>A model-reference is used as identifier for a particular model transfered between systems.
 *
 * @author erik.bamberg@web.de
 */
public class ModelReferenceDTO {
    private String name;
    private String groupId;
    private String artifactId;
    private String version;

    /**
     * constructs a reference.
     *
     * @param name of the model
     * @param groupId of the model
     * @param artifactId of the model
     * @param version of the model
     */
    public ModelReferenceDTO(String name, String groupId, String artifactId, String version) {
        super();
        this.name = name;
        this.groupId = groupId;
        this.artifactId = artifactId;
        this.version = version;
    }

    /**
     * constructs a modelReference from an artifact.
     *
     * <p>throws an IllegalArgumentException of the artifact is not a model.
     *
     * @param artifact to construct a reference from
     */
    public ModelReferenceDTO(Artifact artifact) {
        //       if (!"model".equals(artifact.getMetadata().getResourceType())) {
        //           throw new IllegalArgumentException(
        //                   "trying to build a model reference from an artifact which is not model.
        // resourcetype:"+artifact.getMetadata().getResourceType());
        //       }
        this.name = artifact.getName();
        this.version = artifact.getVersion();
        this.groupId = artifact.getMetadata().getGroupId();
        this.artifactId = artifact.getMetadata().getArtifactId();
    }

    /**
     * get the name of the model.
     *
     * @return name of the model.
     */
    public String getName() {
        return name;
    }
    /**
     * get the group identifier of that model.
     *
     * @return group id of the model.
     */
    public String getGroupId() {
        return groupId;
    }
    /**
     * get the artifact id of that model.
     *
     * @return artifact id of the model.
     */
    public String getArtifactId() {
        return artifactId;
    }

    /**
     * get the version of that model.
     *
     * @return version of the model.
     */
    public String getVersion() {
        return version;
    }
}
