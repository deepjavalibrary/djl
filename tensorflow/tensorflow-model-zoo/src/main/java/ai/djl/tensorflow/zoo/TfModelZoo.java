/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.tensorflow.zoo;

import ai.djl.Application.CV;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.tensorflow.engine.TfEngine;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/** TfModelZoo is a repository that contains the TensorFlow models for DJL. */
public class TfModelZoo extends ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("TensorFlow", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.tensorflow";

    private static final List<ModelLoader> MODEL_LOADERS = new ArrayList<>();

    static {
        MRL resnet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(resnet));

        MRL mobilenet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mobilenet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(mobilenet));

        MRL ssd = REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(ssd));
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
        return Collections.singleton(TfEngine.ENGINE_NAME);
    }
}
