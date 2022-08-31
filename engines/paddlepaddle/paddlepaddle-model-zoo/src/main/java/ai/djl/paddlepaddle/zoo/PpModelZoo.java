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
package ai.djl.paddlepaddle.zoo;

import ai.djl.Application.CV;
import ai.djl.paddlepaddle.engine.PpEngine;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;

import java.util.Collections;
import java.util.Set;

/** PpModelZoo is a repository that contains all PaddlePaddle models for DJL. */
public class PpModelZoo extends ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("Paddle", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.paddlepaddle";

    PpModelZoo() {
        addModel(
                REPOSITORY.model(
                        CV.IMAGE_CLASSIFICATION, GROUP_ID, "mask_classification", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "word_rotation", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "face_detection", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "word_detection", "0.0.1"));
        addModel(REPOSITORY.model(CV.WORD_RECOGNITION, GROUP_ID, "word_recognition", "0.0.1"));
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton(PpEngine.ENGINE_NAME);
    }
}
