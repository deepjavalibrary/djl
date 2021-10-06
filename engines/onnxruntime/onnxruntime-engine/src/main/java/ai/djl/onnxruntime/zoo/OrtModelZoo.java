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

import ai.djl.Application.CV;
import ai.djl.Application.Tabular;
import ai.djl.onnxruntime.engine.OrtEngine;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/** OrtModelZoo is a repository that contains all Onnx models for DJL. */
public class OrtModelZoo extends ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("Ort", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.onnxruntime";

    private static final List<ModelLoader> MODEL_LOADERS = new ArrayList<>();

    static {
        MRL resnet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(resnet));

        MRL irisFlower =
                REPOSITORY.model(Tabular.SOFTMAX_REGRESSION, GROUP_ID, "iris_flowers", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(irisFlower));
    }

    /** {@inheritDoc} */
    @Override
    public List<ModelLoader> getModelLoaders() {
        return MODEL_LOADERS;
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton(OrtEngine.ENGINE_NAME);
    }
}
