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

import ai.djl.modality.cv.zoo.ImageClassificationModelLoader;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.zoo.cv.gan.BigGANModelLoader;
import ai.djl.pytorch.zoo.cv.objectdetection.PtSsdModelLoader;
import ai.djl.pytorch.zoo.nlp.qa.BertQAModelLoader;
import ai.djl.pytorch.zoo.nlp.sentimentanalysis.DistilBertSentimentAnalysisModelLoader;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelZoo;
import java.util.Collections;
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

    public static final ImageClassificationModelLoader RESNET =
            new ImageClassificationModelLoader(REPOSITORY, GROUP_ID, "resnet", "0.0.1", ZOO);
    public static final PtSsdModelLoader SSD =
            new PtSsdModelLoader(REPOSITORY, GROUP_ID, "ssd", "0.0.1", ZOO);

    public static final BertQAModelLoader BERT_QA = new BertQAModelLoader(REPOSITORY);

    public static final DistilBertSentimentAnalysisModelLoader DB_SENTIMENT_ANALYSIS =
            new DistilBertSentimentAnalysisModelLoader(REPOSITORY);

    public static final BigGANModelLoader BIG_GAN = new BigGANModelLoader(REPOSITORY);

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
