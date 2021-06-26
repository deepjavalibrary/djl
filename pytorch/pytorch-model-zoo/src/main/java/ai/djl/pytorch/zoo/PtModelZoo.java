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
import ai.djl.pytorch.engine.PtEngine;
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
 * PtModelZoo is a repository that contains all PyTorch models in {@link
 * ai.djl.pytorch.engine.PtSymbolBlock} for DJL.
 */
public class PtModelZoo implements ModelZoo {

    private static final String DJL_REPO_URL = "https://mlrepo.djl.ai/";
    private static final Repository REPOSITORY = Repository.newInstance("PyTorch", DJL_REPO_URL);
    private static final PtModelZoo ZOO = new PtModelZoo();
    public static final String GROUP_ID = "ai.djl.pytorch";

    private static final List<ModelLoader> MODEL_LOADERS = new ArrayList<>();

    static {
        MRL resnet = MRL.model(CV.IMAGE_CLASSIFICATION, GROUP_ID, "resnet");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, resnet, "0.0.1", ZOO));

        MRL ssd = MRL.model(CV.OBJECT_DETECTION, GROUP_ID, "ssd");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, ssd, "0.0.1", ZOO));

        MRL bertQa = MRL.model(NLP.QUESTION_ANSWER, GROUP_ID, "bertqa");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, bertQa, "0.0.1", ZOO));

        MRL sentimentAnalysis = MRL.model(NLP.SENTIMENT_ANALYSIS, GROUP_ID, "distilbert");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, sentimentAnalysis, "0.0.1", ZOO));

        MRL bigGan = MRL.model(CV.IMAGE_GENERATION, GROUP_ID, "biggan-deep");
        MODEL_LOADERS.add(new BaseModelLoader(REPOSITORY, bigGan, "0.0.1", ZOO));
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
        return Collections.singleton(PtEngine.ENGINE_NAME);
    }
}
