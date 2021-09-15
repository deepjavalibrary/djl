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
public class MxModelZoo extends ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("MXNet", DJL_REPO_URL);
    public static final String GROUP_ID = "ai.djl.mxnet";

    private static final List<ModelLoader> MODEL_LOADERS = new ArrayList<>();

    static {
        MRL ssd = REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(ssd));

        MRL yolo = REPOSITORY.model(CV.OBJECT_DETECTION, GROUP_ID, "yolo", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(yolo));

        MRL alexnet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "alexnet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(alexnet));

        MRL darknet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "darknet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(darknet));

        MRL densenet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "densenet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(densenet));

        MRL googlenet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "googlenet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(googlenet));

        MRL inceptionv3 =
                REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "inceptionv3", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(inceptionv3));

        MRL mlp = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mlp", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(mlp));

        MRL mobilenet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "mobilenet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(mobilenet));

        MRL resnest = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnest", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(resnest));

        MRL resnet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(resnet));

        MRL senet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "senet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(senet));

        MRL seresnext = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "se_resnext", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(seresnext));

        MRL squeezenet = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "squeezenet", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(squeezenet));

        MRL vgg = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "vgg", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(vgg));

        MRL xception = REPOSITORY.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "xception", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(xception));

        MRL simplePose = REPOSITORY.model(CV.POSE_ESTIMATION, GROUP_ID, "simple_pose", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(simplePose));

        MRL maskrcnn = REPOSITORY.model(CV.INSTANCE_SEGMENTATION, GROUP_ID, "mask_rcnn", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(maskrcnn));

        MRL actionRecognition =
                REPOSITORY.model(CV.ACTION_RECOGNITION, GROUP_ID, "action_recognition", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(actionRecognition));

        MRL bertQa = REPOSITORY.model(NLP.QUESTION_ANSWER, GROUP_ID, "bertqa", "0.0.1");
        MODEL_LOADERS.add(new BaseModelLoader(bertQa));

        MRL glove = REPOSITORY.model(NLP.WORD_EMBEDDING, GROUP_ID, "glove", "0.0.2");
        MODEL_LOADERS.add(new BaseModelLoader(glove));
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
