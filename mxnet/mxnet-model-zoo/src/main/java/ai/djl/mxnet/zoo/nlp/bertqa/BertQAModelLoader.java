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
package ai.djl.mxnet.zoo.nlp.bertqa;

import ai.djl.mxnet.zoo.BaseSymbolModelLoader;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.Anchor;
import ai.djl.repository.MRL;
import ai.djl.repository.MRL.Model.NLP;
import ai.djl.repository.Repository;
import ai.djl.translate.Translator;

public class BertQAModelLoader extends BaseSymbolModelLoader<QAInput, String> {

    private static final Anchor BASE_ANCHOR = NLP.BERTQA;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "bertqa";
    private static final String VERSION = "0.0.1";

    public BertQAModelLoader(Repository repository) {
        super(repository, new MRL(BASE_ANCHOR, GROUP_ID, ARTIFACT_ID), VERSION);
    }

    @Override
    public Translator<QAInput, String> getTranslator() {
        return new BertQATranslator();
    }
}
