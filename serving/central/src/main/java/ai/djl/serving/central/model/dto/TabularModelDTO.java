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
 * A ModelDTO with structure data-structures for tabular data models.
 *
 * @author erik.bamberg@web.de
 */
public class TabularModelDTO extends ModelDTO {

    /**
     * creates a ModelDTO for tabular data tasks.
     *
     * @param name of model.
     * @param groupId of model.
     * @param artifactId of model.
     * @param version of model.
     */
    public TabularModelDTO(String name, String groupId, String artifactId, String version) {
        super(name, groupId, artifactId, version);
    }

    /**
     * creates a ModelDTO for tabular data tasks.
     *
     * @param artifact to build this data transfer object.
     */
    public TabularModelDTO(Artifact artifact) {
        super(artifact);
        cleanUpDuplicateProperties();
    }
}
