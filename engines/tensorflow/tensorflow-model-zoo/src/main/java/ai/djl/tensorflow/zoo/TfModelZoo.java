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
import ai.djl.repository.RemoteRepository;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.tensorflow.engine.TfEngine;

import java.util.Collections;
import java.util.Set;

/** TfModelZoo is a repository that contains the TensorFlow models for DJL. */
public class TfModelZoo extends ModelZoo {

    private static final Repository REPOSITORY = new RemoteRepository("TensorFlow", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.tensorflow";

    TfModelZoo() {
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mobilenet", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.1"));
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
