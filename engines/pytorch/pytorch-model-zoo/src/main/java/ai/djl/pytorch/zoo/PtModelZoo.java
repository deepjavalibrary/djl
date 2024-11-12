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
package ai.djl.pytorch.zoo;

import ai.djl.Application.CV;
import ai.djl.Application.NLP;
import ai.djl.Application.TimeSeries;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;

import java.util.Collections;
import java.util.Set;

/**
 * PtModelZoo is a repository that contains all PyTorch models in {@link
 * ai.djl.pytorch.engine.PtSymbolBlock} for DJL.
 */
public class PtModelZoo extends ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("PyTorch", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.pytorch";

    PtModelZoo() {
        addModel(
                REPOSITORY.model(
                        CV.ACTION_RECOGNITION,
                        GROUP_ID,
                        "Human-Action-Recognition-VIT-Base-patch16-224",
                        "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1"));
        addModel(
                REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet18_embedding", "0.0.1"));
        addModel(REPOSITORY.model(CV.INSTANCE_SEGMENTATION, GROUP_ID, "yolo11n-seg", "0.0.1"));
        addModel(REPOSITORY.model(CV.INSTANCE_SEGMENTATION, GROUP_ID, "yolov8n-seg", "0.0.1"));
        addModel(REPOSITORY.model(CV.MASK_GENERATION, GROUP_ID, "sam2-hiera-tiny", "0.0.1"));
        addModel(REPOSITORY.model(CV.MASK_GENERATION, GROUP_ID, "sam2-hiera-large", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "yolo11n", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "yolov5s", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "yolov8n", "0.0.1"));
        addModel(REPOSITORY.model(CV.POSE_ESTIMATION, GROUP_ID, "yolo11n-pose", "0.0.1"));
        addModel(REPOSITORY.model(CV.POSE_ESTIMATION, GROUP_ID, "yolov8n-pose", "0.0.1"));
        addModel(REPOSITORY.model(NLP.QUESTION_ANSWER, GROUP_ID, "bertqa", "0.0.1"));
        addModel(REPOSITORY.model(NLP.SENTIMENT_ANALYSIS, GROUP_ID, "distilbert", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_GENERATION, GROUP_ID, "biggan-deep", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_GENERATION, GROUP_ID, "cyclegan", "0.0.1"));
        addModel(REPOSITORY.model(CV.SEMANTIC_SEGMENTATION, GROUP_ID, "deeplabv3", "0.0.1"));
        addModel(REPOSITORY.model(TimeSeries.FORECASTING, GROUP_ID, "deepar", "0.0.1"));
    }

    /** {@inheritDoc} */
    @Override
    public String getGroupId() {
        return GROUP_ID;
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedEngines() {
        return Collections.singleton(PtEngine.ENGINE_NAME);
    }
}
