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
package ai.djl.mxnet.zoo.nlp.qa;

import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.Anchor;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.NLP;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Translator;

/**
 * Model loader for BERT QA models.
 *
 * <p>The model was trained on Gluon and loaded in DJL in MXNet Symbol Block.
 *
 * <p>See <a href="https://arxiv.org/pdf/1810.04805.pdf">the BERT paper</a> or the <a
 * href="https://github.com/awslabs/djl/blob/master/jupyter/BERTQA.ipynb">jupyter demo</a> for more
 * information about BERT.
 *
 * @see ai.djl.mxnet.engine.MxSymbolBlock
 */
public class BertQAModelLoader extends BaseModelLoader<QAInput, String> {

    private static final Anchor BASE_ANCHOR = NLP.QUESTION_ANSWER;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "bertqa";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public BertQAModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    /** {@inheritDoc} */
    @Override
    public Translator<QAInput, String> getTranslator(Artifact artifact) {
        return new BertQATranslator();
    }
}
