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
package ai.djl.mxnet.zoo;

import ai.djl.Application.CV;
import ai.djl.Application.NLP;
import ai.djl.Application.TimeSeries;
import ai.djl.mxnet.engine.MxEngine;
import ai.djl.repository.RemoteRepository;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;

import java.util.Collections;
import java.util.Set;

/**
 * MxModelZoo is a repository that contains all MXNet models in {@link
 * ai.djl.mxnet.engine.MxSymbolBlock} for DJL.
 */
public class MxModelZoo extends ModelZoo {

    private static final Repository REPOSITORY = new RemoteRepository("MXNet", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.mxnet";

    MxModelZoo() {
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.1"));
        addModel(REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "yolo", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "alexnet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "darknet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "densenet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "googlenet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "inceptionv3", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mlp", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mobilenet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnest", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "senet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "se_resnext", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "squeezenet", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "vgg", "0.0.1"));
        addModel(REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "xception", "0.0.1"));
        addModel(REPOSITORY.model(CV.POSE_ESTIMATION, GROUP_ID, "simple_pose", "0.0.1"));
        addModel(REPOSITORY.model(CV.INSTANCE_SEGMENTATION, GROUP_ID, "mask_rcnn", "0.0.1"));
        addModel(REPOSITORY.model(CV.ACTION_RECOGNITION, GROUP_ID, "action_recognition", "0.0.1"));
        addModel(REPOSITORY.model(NLP.QUESTION_ANSWER, GROUP_ID, "bertqa", "0.0.1"));
        addModel(REPOSITORY.model(NLP.WORD_EMBEDDING, GROUP_ID, "glove", "0.0.2"));
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
        return Collections.singleton(MxEngine.ENGINE_NAME);
    }
}
