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
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.fasttext.FtModel;
import ai.djl.fasttext.zoo.FtModelZoo;
import ai.djl.modality.Classifications;
import ai.djl.nn.Block;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

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
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new FtModelZoo());
        factories.put(new Pair<>(String.class, Classifications.class), new FactoryImpl());
    }

    /**
     * Loads the model with the given search filters.
     *
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<String, Classifications> loadModel()
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<String, Classifications> criteria =
                Criteria.builder().setTypes(String.class, Classifications.class).build();
        return loadModel(criteria);
    }

    /** {@inheritDoc} */
    @Override
    protected Model createModel(
            Path modelPath,
            String name,
            Device device,
            Block block,
            Map<String, Object> arguments,
            String engine) {
        return new FtModel(name);
    }

    private static final class FactoryImpl implements TranslatorFactory<String, Classifications> {

        /** {@inheritDoc} */
        @Override
        public Translator<String, Classifications> newInstance(
                Model model, Map<String, ?> arguments) {
            return null;
        }
    }
}
