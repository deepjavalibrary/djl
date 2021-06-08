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

import ai.djl.Application;
import ai.djl.Application.CV;
import ai.djl.Application.NLP;
import ai.djl.Application.Tabular;
import ai.djl.repository.Artifact;

/**
 * A factory to create ModelDTO from an modelzoo artifact.
 *
 * <p>The created model reflects the artifact-application-type in the ModelDTO hierarchy.
 *
 * @author erik.bamberg@web.de
 */
public class DataTransferObjectFactory {

    /**
     * creates a ModelDTO from an artifact.
     *
     * @param artifact as source
     * @return modelDTO created.
     */
    public ModelDTO create(Artifact artifact) {
        ModelDTO result;
        Application type = artifact.getMetadata().getApplication();
        if (type.matches(CV.ANY)) {
            result = new ComputerVisionModelDTO(artifact);
        } else if (type.matches(NLP.ANY)) {
            result = new NaturalLanguageProcessingModelDTO(artifact);
        } else if (type.matches(Tabular.ANY)) {
            result = new TabularModelDTO(artifact);
        } else {
            result = new ModelDTO(artifact);
        }
        return result;
    }
}
