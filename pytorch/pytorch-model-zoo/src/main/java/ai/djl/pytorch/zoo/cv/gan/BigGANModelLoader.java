/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.pytorch.zoo.cv.gan;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.translator.BigGANTranslator;
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

/** Model loader for BigGAN. */
public class BigGANModelLoader extends BaseModelLoader {

    private static final Application APPLICATION = Application.CV.GAN;
    private static final String GROUP_ID = PtModelZoo.GROUP_ID;
    private static final String ARTIFACT_ID = "biggan-deep";
    private static final String VERSION = "0.0.1";

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     */
    public BigGANModelLoader(Repository repository) {
        super(repository, MRL.model(APPLICATION, GROUP_ID, ARTIFACT_ID), VERSION, new PtModelZoo());
        FactoryImpl factory = new FactoryImpl();
        factories.put(new Pair<>(int[].class, Image[].class), factory);
    }

    /**
     * Loads the model.
     *
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<int[], Image[]> loadModel()
            throws MalformedModelException, ModelNotFoundException, IOException {
        return loadModel(null, null, null);
    }

    /**
     * Loads the model.
     *
     * @param progress the progress tracker to update while loading the model
     * @return the loaded model
     * @throws IOException for various exceptions loading data from the repository
     * @throws ModelNotFoundException if no model with the specified criteria is found
     * @throws MalformedModelException if the model data is malformed
     */
    public ZooModel<int[], Image[]> loadModel(Progress progress)
            throws MalformedModelException, ModelNotFoundException, IOException {
        return loadModel(null, null, progress);
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
    public ZooModel<int[], Image[]> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<int[], Image[]> criteria =
                Criteria.builder()
                        .setTypes(int[].class, Image[].class)
                        .optModelZoo(modelZoo)
                        .optGroupId(resource.getMrl().getGroupId())
                        .optArtifactId(resource.getMrl().getArtifactId())
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<int[], Image[]> {

        /** {@inheritDoc} */
        @Override
        public Translator<int[], Image[]> newInstance(Model model, Map<String, ?> arguments) {
            Float truncation = (Float) arguments.get("truncation");
            return new BigGANTranslator(truncation == null ? 0.5f : truncation);
        }
    }
}
