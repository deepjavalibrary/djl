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
package ai.djl.onnxruntime.zoo.tabular.randomforest;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.onnxruntime.zoo.OrtModelZoo;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/** Model loader for onnx iris_flowers models. */
public class IrisClassificationModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.Tabular.RANDOM_FOREST;
    private static final String GROUP_ID = OrtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "iris_flowers";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public IrisClassificationModelLoader(Repository repository) {
        super(
                repository,
                MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID),
                VERSION,
                new OrtModelZoo());
        factories.put(new Pair<>(IrisFlower.class, Classifications.class), new FactoryImpl());
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

    private static final class FactoryImpl
            implements TranslatorFactory<IrisFlower, Classifications> {

        /** {@inheritDoc} */
        @Override
        public Translator<IrisFlower, Classifications> newInstance(
                Model model, Map<String, ?> arguments) {
            return new IrisTranslator();
        }
    }

    private static final class IrisTranslator implements Translator<IrisFlower, Classifications> {

        private List<String> synset;

        public IrisTranslator() {
            // species name
            synset = Arrays.asList("setosa", "versicolor", "virginica");
        }

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, IrisFlower input) {
            float[] data = {
                input.getSepalLength(),
                input.getSepalWidth(),
                input.getPetalLength(),
                input.getPetalWidth()
            };
            NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
            return new NDList(array);
        }

        /** {@inheritDoc} */
        @Override
        public Classifications processOutput(TranslatorContext ctx, NDList list) {
            return new Classifications(synset, list.get(1));
        }

        /** {@inheritDoc} */
        @Override
        public Batchifier getBatchifier() {
            return null;
        }
    }
}
