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
package ai.djl.fasttext.zoo.nlp.textclassification;

import ai.djl.Application;
import ai.djl.fasttext.engine.TextClassificationTranslator;
import ai.djl.fasttext.zoo.FtModelZoo;
import ai.djl.modality.Classifications;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import java.lang.reflect.Type;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Model loader for fastText cooking stackexchange models. */
public class TextClassificationModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.NLP.TEXT_CLASSIFICATION;
    private static final String GROUP_ID = FtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "cooking_stackexchange";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public TextClassificationModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        Map<Type, TranslatorFactory<?, ?>> map = new ConcurrentHashMap<>();
        map.put(Classifications.class, new FactoryImpl());
        factories.put(String.class, map);
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    private static final class FactoryImpl implements TranslatorFactory<String, Classifications> {

        @Override
        public Translator<String, Classifications> newInstance(Map<String, Object> arguments) {
            return new TextClassificationTranslator();
        }
    }
}
