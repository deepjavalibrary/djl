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
package ai.djl.mxnet.zoo.nlp.embedding;

import ai.djl.Application;
import ai.djl.Application.NLP;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.modality.nlp.embedding.WordEmbedding;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.core.Embedding;
import ai.djl.repository.Artifact;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import ai.djl.util.Utils;
import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * A {@link ai.djl.repository.zoo.ModelLoader} for a {@link WordEmbedding} based on <a
 * href="https://nlp.stanford.edu/projects/glove/">GloVe</a>.
 */
public class GloveWordEmbeddingModelLoader extends BaseModelLoader<NDList, NDList> {

    private static final Application APPLICATION = NLP.WORD_EMBEDDING;
    private static final String GROUP_ID = MxModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "glove";
    private static final String VERSION = "0.0.1";

    /**
     * Constructs a {@link GloveWordEmbeddingModelLoader} given the repository.
     *
     * @param repository the repository to load the model from
     */
    public GloveWordEmbeddingModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION);
        factories.put(new Pair<>(String.class, NDList.class), new FactoryImpl());
    }

    /** {@inheritDoc} */
    @Override
    public Application getApplication() {
        return APPLICATION;
    }

    /** {@inheritDoc} */
    @Override
    protected Model createModel(Device device, Artifact artifact, Map<String, Object> arguments)
            throws IOException {
        Model model = Model.newInstance(device);
        List<String> idxToToken =
                Utils.readLines(
                        repository.openStream(artifact.getFiles().get("idx_to_token"), null));
        TrainableWordEmbedding wordEmbedding =
                TrainableWordEmbedding.builder()
                        .setEmbeddingSize(
                                Integer.parseInt(artifact.getProperties().get("dimensions")))
                        .setItems(idxToToken)
                        .optUnknownToken((String) arguments.get("unknownToken"))
                        .optUseDefault(true)
                        .optSparseGrad(false)
                        .build();
        model.setBlock(wordEmbedding);
        model.setProperty("unknownToken", (String) arguments.get("unknownToken"));
        return model;
    }

    /** {@inheritDoc} */
    @Override
    public ZooModel<NDList, NDList> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optApplication(NLP.WORD_EMBEDDING)
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<String, NDList> {

        /** {@inheritDoc} */
        @Override
        public Translator<String, NDList> newInstance(Map<String, Object> arguments) {
            String unknownToken = (String) arguments.get("unknownToken");
            return new TranslatorImpl(unknownToken);
        }
    }

    private static final class TranslatorImpl implements Translator<String, NDList> {

        private String unknownToken;
        private Embedding<String> embedding;

        public TranslatorImpl(String unknownToken) {
            this.unknownToken = unknownToken;
        }

        /** {@inheritDoc} */
        @Override
        @SuppressWarnings("unchecked")
        public void prepare(NDManager manager, Model model) {
            try {
                embedding = (Embedding<String>) model.getBlock();
            } catch (ClassCastException e) {
                throw new IllegalArgumentException("The model was not an embedding", e);
            }
        }

        /** {@inheritDoc} */
        @Override
        public NDList processOutput(TranslatorContext ctx, NDList list) {
            return list;
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, String input) {
            if (embedding.hasItem(input)) {
                return new NDList(ctx.getNDManager().create(embedding.embed(input)));
            } else {
                return new NDList(ctx.getNDManager().create(embedding.embed(unknownToken)));
            }
        }
    }
}
