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

import ai.djl.modality.cv.zoo.ImageClassificationModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.tensorflow.engine.TfEngine;
import ai.djl.tensorflow.zoo.cv.objectdetction.TfSsdModelLoader;
import java.util.Collections;
import java.util.Set;

/**
 * TfModelZoo is a repository that contains all TensorFlow models in {@link
 * ai.djl.tensorflow.engine.TfSymbolBlock} for DJL.
 */
public class TfModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("TensorFlow", DJL_REPO_URL);
    private static final TfModelZoo ZOO = new TfModelZoo();
    public static final String GROUP_ID = "ai.djl.tensorflow";

    public static final ImageClassificationModelLoader RESNET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "resnet", "0.0.1", ZOO);
    public static final ImageClassificationModelLoader MOBILENET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "mobilenet", "0.0.1", ZOO);
    public static final TfSsdModelLoader SSD =
            new TfSsdModelLoader(REPOSITORY, GROUP_ID, "ssd", "0.0.1", ZOO);

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
