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

import ai.djl.paddlepaddle.engine.PpEngine;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import java.util.Collections;
import java.util.Set;

/** PpModelZoo is a repository that contains all PaddlePaddle models for DJL. */
public class PpModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("Paddle", DJL_REPO_URL);
    private static final PpModelZoo ZOO = new PpModelZoo();
    public static final String GROUP_ID = "ai.djl.paddlepaddle";

    public static final PpFaceDetection FACE_DETECTION =
            new PpFaceDetection(REPOSITORY, GROUP_ID, "face_detection", "0.0.1", ZOO);

    public static final PpMaskClassification MASK_DETECTION =
            new PpMaskClassification(REPOSITORY, GROUP_ID, "mask_classification", "0.0.1", ZOO);

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
