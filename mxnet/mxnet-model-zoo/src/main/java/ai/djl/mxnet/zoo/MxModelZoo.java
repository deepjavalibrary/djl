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
import ai.djl.mxnet.engine.MxEngine;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.ModelLoader;
import ai.djl.repository.zoo.ModelZoo;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * MxModelZoo is a repository that contains all MXNet models in {@link
 * ai.djl.mxnet.engine.MxSymbolBlock} for DJL.
 */
public class MxModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("MXNet", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.mxnet";
    private static final MxModelZoo ZOO = new MxModelZoo();

    private static final List<ModelLoader> MODEL_LOADERS = new ArrayList<>();

    static {
        MRL ssd = MRL.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, ssd, "0.0.1", ZOO));

        MRL yolo = MRL.model(CV.OBJECT_DETECTION, GROUP_ID, "yolo");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, yolo, "0.0.1", ZOO));

        MRL alexnet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "alexnet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, alexnet, "0.0.1", ZOO));

        MRL darknet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "darknet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, darknet, "0.0.1", ZOO));

        MRL densenet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "densenet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, densenet, "0.0.1", ZOO));

        MRL googlenet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "googlenet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, googlenet, "0.0.1", ZOO));

        MRL inceptionv3 = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "inceptionv3");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, inceptionv3, "0.0.1", ZOO));

        MRL mlp = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mlp");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, mlp, "0.0.1", ZOO));

        MRL mobilenet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mobilenet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, mobilenet, "0.0.1", ZOO));

        MRL resnest = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnest");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, resnest, "0.0.1", ZOO));

        MRL resnet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, resnet, "0.0.1", ZOO));

        MRL senet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "senet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, senet, "0.0.1", ZOO));

        MRL seresnext = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "se_resnext");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, seresnext, "0.0.1", ZOO));

        MRL squeezenet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "squeezenet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, squeezenet, "0.0.1", ZOO));

        MRL vgg = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "vgg");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, vgg, "0.0.1", ZOO));

        MRL xception = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "xception");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, xception, "0.0.1", ZOO));

        MRL simplePose = MRL.model(CV.POSE_ESTIMATION, GROUP_ID, "simple_pose");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, simplePose, "0.0.1", ZOO));

        MRL maskrcnn = MRL.model(CV.INSTANCE_SEGMENTATION, GROUP_ID, "mask_rcnn");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, maskrcnn, "0.0.1", ZOO));

        MRL actionRecognition = MRL.model(CV.ACTION_RECOGNITION, GROUP_ID, "action_recognition");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, actionRecognition, "0.0.1", ZOO));

        MRL bertQa = MRL.model(NLP.QUESTION_ANSWER, GROUP_ID, "bertqa");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, bertQa, "0.0.1", ZOO));

        MRL glove = MRL.model(NLP.WORD_EMBEDDING, GROUP_ID, "glove");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, glove, "0.0.2", ZOO));
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
        return Collections.singleton(MxEngine.ENGINE_NAME);
    }
}
