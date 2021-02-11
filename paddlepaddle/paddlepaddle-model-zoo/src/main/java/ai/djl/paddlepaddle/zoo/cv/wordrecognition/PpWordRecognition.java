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

package ai.djl.paddlepaddle.zoo.cv.wordrecognition;

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.BaseModelLoader;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import java.io.IOException;
import java.util.Map;

/** Model loader for Word Recognition models. */
public class PpWordRecognition extends BaseModelLoader {

    private static final Application APPLICATION = Application.CV.WORD_RECOGNITION;

    /**
     * Creates the Model loader from the given repository.
     *
     * @param repository the repository to load the model from
     * @param groupId the group id of the model
     * @param artifactId the artifact id of the model
     * @param version the version of the model to load
     * @param modelZoo the modelZoo type that is being used to get supported engine types
     */
    public PpWordRecognition(
            Repository repository,
            String groupId,
            String artifactId,
            String version,
            ModelZoo modelZoo) {
        super(repository, MRL.model(APPLICATION, groupId, artifactId), version, modelZoo);
        FactoryImpl factory = new FactoryImpl();

        factories.put(new Pair<>(Image.class, String.class), factory);
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
    public ZooModel<Image, String> loadModel(
            Map<String, String> filters, Device device, Progress progress)
            throws IOException, ModelNotFoundException, MalformedModelException {
        Criteria<Image, String> criteria =
                Criteria.builder()
                        .setTypes(Image.class, String.class)
                        .optModelZoo(modelZoo)
                        .optGroupId(resource.getMrl().getGroupId())
                        .optArtifactId(resource.getMrl().getArtifactId())
                        .optFilters(filters)
                        .optDevice(device)
                        .optProgress(progress)
                        .build();
        return loadModel(criteria);
    }

    private static final class FactoryImpl implements TranslatorFactory<Image, String> {

        /** {@inheritDoc} */
        @Override
        public Translator<Image, String> newInstance(Model model, Map<String, ?> arguments) {
            return new PpWordRecognitionTranslator();
        }
    }
}
