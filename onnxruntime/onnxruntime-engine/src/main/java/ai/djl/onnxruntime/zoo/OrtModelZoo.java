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
package ai.djl.onnxruntime.zoo;

import ai.djl.engine.Engine;
import ai.djl.onnxruntime.engine.OrtEngine;
import ai.djl.onnxruntime.zoo.tabular.randomforest.IrisClassificationModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import java.util.HashSet;
import java.util.Set;

/** FtModelZoo is a repository that contains all fastText models for DJL. */
public class OrtModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("Ort", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.onnxruntime";

    public static final IrisClassificationModelLoader IRIS_FLOWER =
            new IrisClassificationModelLoader(REPOSITORY);

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        Set<String> set = new HashSet<>();
        set.add(Engine.getInstance().getEngineName());
        set.add(OrtEngine.ENGINE_NAME);
        return set;
    }
}
