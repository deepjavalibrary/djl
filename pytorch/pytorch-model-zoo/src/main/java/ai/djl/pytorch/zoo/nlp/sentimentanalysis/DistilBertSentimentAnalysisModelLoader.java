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
package ai.djl.pytorch.zoo.nlp.sentimentanalysis;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.pytorch.zoo.PtModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.Map;

/**
 * Model loader for PyTorch DistilBERT Sentiment Analysis models.
 *
 * <p>The model was trained with PyTorch from https://github.com/huggingface/transformers.
 *
 * <p>See <a href="https://arxiv.org/abs/1910.01108">the BERT paper</a>
 *
 * @see ai.djl.pytorch.engine.PtSymbolBlock
 */
public class DistilBertSentimentAnalysisModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.NLP.SENTIMENT_ANALYSIS;
    private static final String GROUP_ID = PtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "distilbert";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public DistilBertSentimentAnalysisModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new PtModelZoo());
        factories.put(new Pair<>(String.class, Classifications.class), new FactoryImpl());
    }

    /**
     * Loads the model with the given search filters.
     *
     * @param filters the search filters to match against the loaded model
     * @param device the device the loaded model should use
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<String, Classifications> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<String, Classifications> criteria =
                Criteria.builder()
                        .setTypes(String.class, Classifications.class)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<String, Classifications> {

        /** {@inheritDoc} */
        @Override
        public Translator<String, Classifications> newInstance(
                Model model, Map<String, Object> arguments) {
            return new PtDistilBertTranslator();
        }
    }
}
